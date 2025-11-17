from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit.primitives import StatevectorSampler
#Using qiskit circuit
from qiskit.circuit.library import PauliEvolutionGate
#for transpiler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
#fake hardware
from qiskit.providers.fake_provider import GenericBackendV2
#quantum hardware
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler, EstimatorV2 as Estimator

import numpy as np
import json
from scipy.optimize import minimize


# auxiliary function to load api key
def load_api_key(api_path="./ibm_api.json"):
    """Load API key from JSON file"""
    try:
        with open(api_path, 'r') as f:
            config = json.load(f)
            api_key = config.get("api_key", "")
            cnr = config.get("cnr", "")

    except:
        print("Error loading API key from file.")
        exit()

    return api_key, cnr
    
# auxiliary function to sample most likely bitstring
def int_to_spins(x, n_qubits, spin_sign=1):
    binary = np.binary_repr(x, width=n_qubits)
    if spin_sign==-1:
        #no need to convert if x = (1-z)/2
        spins = [1-2*int(digit) for digit in binary]
    else:
        #convert spin to bit if x = (1+z)/2
        spins = [2*((int(digit)+1)%2)-1 for digit in binary]
    #reverse order (in qiskit last is first)
    spins.reverse()
    return spins


# auxiliary function to set error suppression on quantum hardware
def set_error_suppression(sampler):
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

#auxiliary 2-qubits rotation gate (ITEMC)
def RZYYZ(theta0: Parameter, theta1: Parameter):
    theta1 /= theta0
    operator0 = SparsePauliOp("Z") ^ SparsePauliOp("Y")
    operator1 = SparsePauliOp("Y") ^ SparsePauliOp("Z")
    gate = PauliEvolutionGate(operator0 + theta1*operator1, time=theta0/2)
    return gate

class ITEMCSampler:
    def __init__(self, backend_type="fake", reps=2, tau=0.1, n_shots=50, annealing_time=0.1, n_qubits=30):
        #init qiskit backend
        if backend_type == "fake":
            #fake backend
            coupling_map = [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
            backend = GenericBackendV2(num_qubits=n_qubits, coupling_map=coupling_map)
        else:
            #real backend
            api_key, cnr = load_api_key()
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key, instance=cnr)
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

        #init sampler
        #use session to lock access to quantum hardware and summit iterative jobs
        #(faster but not available on Open plan)
        #session = Session(backend=backend)
        #sampler = Sampler(mode=session)
        #access backend without session: jobs are added to the backend queue (slower)
        sampler = Sampler(mode=backend)
        # Set simple error suppression/mitigation options on quantum hardware
        if backend_type != "fake":
            set_error_suppression(sampler)

        #init gate based solvers
        self.solver = ITEMC_algo(sampler=sampler, pm=pm, reps=reps, tau=tau, n_shots=n_shots)

    def sample(self, h, J):
        #build circuit
        self.solver.build_circuit(h, J)
        #run optimization algorithm
        samples = self.solver.run_quantum_circuit()
        #filter best sample
        #best_sample = self.check_constraint(samples, N_phases)
        return samples[0]


class ITEMC_algo:
    def __init__(self, sampler, pm, reps=2, tau=0.1, n_shots = 100):
        #sign of spin variable z when converting from binary x: x=(1+z)/2 or x=(1-z)/2
        self.spin_sign=-1
        self.reps = reps
        self.tau = tau
        self.n_shots = n_shots
        self.sampler = sampler
        self.pm = pm

    def run_quantum_circuit(self):
        job = self.sampler.run([self.tp_circuit], shots=self.n_shots)
        result = job.result()
        states = result[0].data.meas.get_int_counts()
        ordered_states=sorted(states.items(), key=lambda item: item[1], reverse=True)
        samples=[int_to_spins(state[0], self.n_qubits, self.spin_sign) for state in ordered_states]
        return samples

    def build_circuit(self, h, J):
        self.n_qubits = len(h)
        #compute ITE Hamiltonian parameters theta_i, theta_ij
        theta_i, theta_ij = {}, {}
        for i in range(len(h)):
            #linear terms
            B=h[i]
            theta_i[i]=2*np.arctan(np.tanh(self.tau*B))
            #theta_i[i]=2*np.arctan(-np.exp(-2*self.tau*B))+np.pi/2
            for j in range(i):
                #quadratic terms
                if J[i,j]!=0:
                    theta_ij[(i,j)]=self.get_theta_ij(theta_i[i], theta_i[j], 2*J[i,j]*self.tau)
        #order quadratic terms in ascending order
        theta_ij = dict(sorted(theta_ij.items(), key=lambda item: item[1][0]**2+item[1][1]**2, reverse=True))
        #order quadratic terms so that independent pairs are applied first
        #[(i,j), (l,k),...] with i!=l, j!=k
        theta_ij_ind, theta_ij_dep = {}, {}
        for (i, j), theta in theta_ij.items():
            if any(i in ij for ij in theta_ij_ind.keys()) or any(j in ij for ij in theta_ij_ind.keys()):
                theta_ij_dep[(i,j)]=theta
            else:
                theta_ij_ind[(i,j)]=theta
        #init quantum circuit
        register = QuantumRegister(self.n_qubits)
        self.circuit = QuantumCircuit(register)
        #Apply Hadamard gate to create an equal superposition of all configuration states
        self.circuit.h(register)
        #Apply rotation gates to mimic ITE of Hamiltonian linear terms
        for i, theta in theta_i.items():
            self.circuit.ry(theta, i)
        #Apply 2-qubits rotation gates to mimic ITE of Hamiltonian quadratic terms
        #independent first
        for (i,j), theta in theta_ij_ind.items():
            self.circuit.append(RZYYZ(theta[0], theta[1]), [i,j])
        #dependent last
        for (i,j), theta in theta_ij_dep.items():
            self.circuit.append(RZYYZ(theta[0], theta[1]), [i,j])
        #add measurement
        self.circuit.measure_all()
        #transpile circuit for quantum hardware
        self.tp_circuit = self.pm.run(self.circuit)
        #import matplotlib.pyplot as plt
        #self.circuit.draw("mpl")
        #plt.show()
        #plt.savefig('circuit.pdf')

    def get_theta_ij(self, t_i, t_j, t_ij):
        a=np.cosh(t_ij)-np.sinh(t_ij)*np.sin(t_i)*np.sin(t_j)
        b=np.sinh(t_ij)*np.cos(t_i)
        c=np.sinh(t_ij)*np.cos(t_j)
        d=np.cosh(t_ij)*np.cos(t_i)*np.cos(t_j)
        #overlap function to be maximized
        def f(x):
            f=np.cos(x[0]/2) * (a*np.cos(x[1]/2) + c*np.sin(x[1]/2)) + np.sin(x[0]/2) * (b*np.cos(x[1]/2) - d*np.sin(x[1]/2))
            return -f
        #optimize
        x0 = [0,0]
        result = minimize(f, x0, method='COBYLA')
        return result.x
