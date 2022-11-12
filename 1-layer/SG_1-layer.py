import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------

#dataset path
folder="../datasets/"
filename="housing.csv"
#import dataset file
dataset = np.genfromtxt(folder+filename, delimiter=',')

#shorten dataset
#dataset=dataset[:1500]

#split dataset
X=dataset[:,:-1]
Y=dataset[:,-1]
Y=np.reshape(Y,(-1,1))

#train test
n_test=5000
X_train=X[:-n_test]
Y_train=Y[:-n_test]
X_test=X[-n_test:]
Y_test=Y[-n_test:]

#--------------------------------------

#activation function
def leakyRelu(state, alpha=0.1):
    return state if state >= 0 else alpha*state
#activation function first derivative
def leakyRelu1(state, alpha=0.1):
    return 1 if state >= 0 else alpha
#activation function inverse
def leakyReluInv(state, alpha=0.1):
    return state if state >= 0 else state/alpha
    
#--------------------------------------

#Convert network parameters to spin glass parameters
#Magnetization
def J(x):
    Q=x.shape[0]
    dim_x=x.shape[1]
    J=np.array([[np.sum(x[:,i]*x[:,j], axis=0) for i in range(dim_x)] for j in range(dim_x)])/Q
    return J
#Magnetic field
def h(x,y):
    Q=x.shape[0]
    dim_x=x.shape[1]
    dim_y=y.shape[1]
    h=np.array([[np.sum(-2*y[:,i]*x[:,j], axis=0) for i in range(dim_y)] for j in range(dim_x)])/Q
    return h
#Ground Energy
def E0(y):
    Q=y.shape[0]
    dim_y=y.shape[1]
    E0=np.array([np.sum(y[:,i]*y[:,i], axis=0) for i in range(dim_y)])/Q
    return E0
#Inverse activation function
def f1(y):
    Q=y.shape[0]
    dim_y=y.shape[1]
    f1=np.array([[leakyReluInv(y[q,i]) for i in range(dim_y)] for q in range(Q)])
    return f1


#Spin glass train/test parameters

#Magnetization
J_train=J(X_train)
J_test=J(X_test)
#Magnetic field
h_train=h(X_train,f1(Y_train))
h_test=h(X_test,f1(Y_test))
#Energy
E0_train=E0(f1(Y_train))
E0_test=E0(f1(Y_test))



#--------------------------------------
#Metropolis Algorithm

def Metropolis_step(J, h, S, beta):

    #Select a spin to update at random
    dim_0=S.shape[0]
    dim_1=S.shape[1]
    ind = [np.random.choice(dim_0, 1)[0], np.random.choice(dim_1, 1)[0]]
    
    #Get a new spin value
    S_old=S[ind[0], ind[1]]
    S_new=np.random.uniform(-1,1)
    
    #Calculate energy change due to new spin value
    dE = (S_new -S_old)*(2*np.dot(np.delete(J[ind[0]], ind[0]), np.delete(S[:,ind[1]], ind[0]))+h[ind[0],ind[1]])+J[ind[0], ind[0]]*(S_new**2-S_old**2)
    
    #Check if site should be updated
    if dE <= 0 or np.random.random() < np.exp(-beta*dE):
        S[ind[0], ind[1]] = S_new
    else:
        dE=0
    
    return S, dE
    
    
#--------------------------------------

#Energy function
def Energy(J, h, E0, S):
    E_v=[np.linalg.multi_dot([S[:,i],J, S[:,i]])+np.dot(h[:,i],S[:,i]) +E0[i] for i in range(len(E0))]
    return sum(E_v)


#--------------------------------------




#Main Annealing Metropolis

#model dimensions
input_dim=X_train.shape[1]
output_dim=Y_train.shape[1]

#initial random spin configuration
S = np.random.rand(input_dim,output_dim)*2-1
    
#initial beta (beta = 1/T)
beta_in=0.4
beta_final=5
beta=beta_in
Nsteps=1000
alpha=2
db=(beta_final - beta_in)/Nsteps
E_train=np.zeros(Nsteps)
E_test=np.zeros(Nsteps)
E0 = Energy(J_train, h_train, E0_train, S)

#main loop
for n in range(Nsteps):
    #reduce temperature every step
    beta+=db
    #reset temperature when system is stuck in local minima
    if beta > 1 and E_train[n] > 100:
        beta=beta_in
    #save Energy
    E_train[n] = E_train[n-1] + dE if n > 0 else E0
    E_test[n] = Energy(J_test, h_test, E0_test, S)
    #update spin network
    dE=0
    for k in range(S.shape[0]*S.shape[1]):
        S, dE_s = Metropolis_step(J_train, h_train, S, beta)
        dE+=dE_s
    
E_train=np.reshape(E_train,(-1,1))
E_test=np.reshape(E_test,(-1,1))
loss=np.concatenate((E_train,E_test),axis=1)
    
np.savetxt("log2.csv", np.array(loss), delimiter=",")

#Evaluate reciprocal neural network
w=S
out=np.zeros((len(X_test),output_dim))
for a in range(len(X_test)):
    for i in range(output_dim):
        out[a,i]=leakyRelu(sum(w[j,i]*X_test[a,j] for j in range(input_dim)))
for i in range(10):
    print("P: "+str(out[i])+" Y: "+str(Y_test[i]))
#loss
L=sum(np.square(out-Y_test))/(len(Y_test)*output_dim)
print("Network loss on test dataset: ",L)
    
# plot energy
plt.plot(E_train, color='red')
plt.plot(E_test, color='blue', linestyle='dotted')
plt.show()















