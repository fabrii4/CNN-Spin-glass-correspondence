from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, Parameter
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
import torch.multiprocessing as mp
from itertools import repeat
from torch import seed


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
    
# auxiliary function to sample most likely bitstring
def bit_to_spins(x, spin_sign=1):
    if spin_sign==-1:
        #no need to convert if x = (1-z)/2
        spin = 1-2*int(x)
    else:
        #convert spin to bit if x = (1+z)/2
        spin = 2*((int(x)+1)%2)-1
    return spin


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
        self.solver = ITEMC_algo(sampler=sampler, pm=pm, tau=tau, n_shots=n_shots)

    def sample(self, h, J):
        n_qubits_max=8
        n_qubits=n_qubits_max-1
        indices_list=[]
        indices=[]
        for i in range(len(h)):
            if len(indices)==n_qubits:
                indices_list.append(indices)
                indices=[]
            indices.append(i)
        if len(indices)>0:
            indices_list.append(indices)

        #calculate angles to be used in rotation matrices
        theta_i, theta_ij = {}, {}
        h_max=max(abs(h))
        J_max=max(abs(J.flatten()))
        #scale tau
        tau=self.solver.tau/h_max
        tau_J=self.solver.tau/J_max
        #exit()
        for i in (range(len(h))):
            theta_i[i]=2*np.arctan(np.tanh(tau*h[i]))
            for j in range(len(h)):
                if j!=i:
                    #linear terms
                    theta_i[j]=2*np.arctan(np.tanh(tau*h[j]))
                    #quadratic terms
                    if J[i,j]!=0:
                        theta_ij[(i,j)]=self.solver.get_theta_ij(theta_i[i], theta_i[j], J[i,j]*tau_J)
        #build circuit
        samples=[0]*len(h)            
        for indices in indices_list:
            #p=mp.Process(target=self.sample_i, args=(h, J, samples, i))
            #p.daemon = False
            #p.start()
            #p.join()
            self.sample_i(theta_i, theta_ij, samples, indices)
                
        return samples
        
    #sample a single qubit    
    def sample_i(self, h, J, samples, indices):
        self.solver.build_circuit(h, J, indices)
        #run optimization algorithm
        samples_i = self.solver.run_quantum_circuit()
        #filter best sample
        for i, idx in enumerate(indices):
            samples[idx]=samples_i[0][i]


class ITEMC_algo:
    def __init__(self, sampler, pm, tau=0.1, n_shots = 100):
        #sign of spin variable z when converting from binary x: x=(1+z)/2 or x=(1-z)/2
        self.spin_sign=-1
        self.tau = tau
        self.n_shots = n_shots
        self.sampler = sampler
        self.pm = pm

#    def run_quantum_circuit(self):
#        job = self.sampler.run([self.tp_circuit], shots=self.n_shots)
#        result = job.result()
#        states = result[0].data.c.get_counts()
#        ordered_states=sorted(states.items(), key=lambda item: item[1], reverse=True)
#        samples=[bit_to_spins(state[0], self.spin_sign) for state in ordered_states]
#        return samples
        
    def run_quantum_circuit(self):
        job = self.sampler.run([self.tp_circuit], shots=self.n_shots)
        result = job.result()
        #states = result[0].data.meas.get_int_counts()
        states = result[0].data.c.get_int_counts()
        ordered_states=sorted(states.items(), key=lambda item: item[1], reverse=True)
        samples=[int_to_spins(state[0], self.n_qubits, self.spin_sign) for state in ordered_states]
        return samples

    def build_circuit(self, theta_i, theta_ij, indices):
        self.n_qubits = len(indices)
#        #compute ITE Hamiltonian parameters theta_i, theta_ij
#        theta_i, theta_ij = {}, {}
#        for i in indices:
#            theta_i[i]=2*np.arctan(np.tanh(self.tau*h[i]))
#            for j in range(len(h)):
#                if j!=i:
#                    #linear terms
#                    theta_i[j]=2*np.arctan(np.tanh(self.tau*h[j]))
#                    #quadratic terms
#                    if J[i,j]!=0:
#                        theta_ij[(i,j)]=self.get_theta_ij(theta_i[i], theta_i[j], J[i,j]*self.tau)
        #init quantum circuit
        register = QuantumRegister(self.n_qubits)
        ancillas = AncillaRegister(1)
        classical = ClassicalRegister(self.n_qubits, 'c')
        self.circuit = QuantumCircuit(register, ancillas, classical)
        #Apply Hadamard gate to create an equal superposition of all configuration states
        self.circuit.h(register)
        #Apply 1-qubit rotation gate on target qubit
        for i in range(self.n_qubits):
            i_q=indices[i]
            i_a=self.n_qubits
            if theta_i[i_q]!=0:
                self.circuit.ry(theta_i[i_q], i)
            for j in range(len(theta_i)):
                if j!=i_q:
                    #reset ancilla qubits
                    self.circuit.reset(i_a)
                    #apply Hadamard on acillas qubits
                    self.circuit.h(i_a)
                    #Apply 1-qubit rotation gates on ancillas qubits
                    if theta_i[j]!=0:
                        self.circuit.ry(theta_i[j], i_a)
                    #Apply 2-qubits rotation gates between target and all other qubits
                    if sum(theta_ij[(i_q,j)])!=0:
                        theta = theta_ij[(i_q,j)]
                        self.circuit.append(RZYYZ(theta[0], theta[1]), [i,i_a])
        #add measurement
        self.circuit.measure(register, classical)
        #transpile circuit for quantum hardware
        self.tp_circuit = self.pm.run(self.circuit)
        #import matplotlib.pyplot as plt
        #self.circuit.draw(output="mpl", filename="circuit-mpl.jpeg")
        #exit()

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
