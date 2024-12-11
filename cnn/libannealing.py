import torch
import torch.nn as nn
import tqdm
import json
import numpy as np
from opt_einsum import contract
import torch.nn.functional as F
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite





###################################
#convert to binary representation
def dec2bin(x, bits=8):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    
def bin2dec(b, bits=8):
    mask = 2**torch.arange(bits).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

###################################
#Neural to spin representation conversion

#weight to spin configuration  
def weight_to_spin(c, c_orig, L, nbit):
#    Nb=2**nbit-1
#    c=(((c+L-c_orig)/(2*L))*Nb).round().long()
#    c=dec2bin(c, bits=nbit)
#    c=c.flatten(1,-1)
#    c_s=2*c.long()-1
    #random configuration
    c_s=2*torch.randint(2,(c.shape[0],c.flatten(1,-1).shape[1]*nbit))-1
    return c_s
    
#spin configuration to weight
def spin_to_weight(c_s, c_orig, L, nbit):
    Nb=2**nbit-1
    shape=torch.cat((torch.tensor(c_orig.shape),torch.tensor([nbit])))
    c_s=c_s.reshape(*shape)
    c=bin2dec(c_s, bits=nbit)
    c=L/Nb*c+c_orig
    return nn.Parameter(c)


#conv layer
def conv_layer_to_ising(s, f1z, c, K, L, nbit=8):
    Q=s.shape[0]
    N2=(s.shape[-1]-K+1)**2
    F=1/(Q*N2)
    Nb=2**nbit-1
    #unfold x for convolution (x_{c(i-k)(j-l)})
    s_patch= s.unfold(2, K, 1).unfold(3, K, 1)
    #binary tensor 
    t_bin=torch.tensor([2**e for e in range(nbit)]).to(s_patch.device)
    #J
    J_orig=F*contract('abijkl,acijmn->bklcmn', s_patch, s_patch)
    J=(L/Nb)**2*contract('bklcmn,q,w->bklqcmnw', J_orig, t_bin, t_bin)
    J_s=J.flatten(4,-1).flatten(0,-2)
    #h
    h_orig=-2*F*contract('abijkl,acij->bckl', s_patch, f1z)
    h=h_orig+2*contract('cklbmn,abmn->cakl', J_orig, c)
    h=L/Nb*contract('bckl,q->cbklq', h, t_bin)
    h_s=h.flatten(1,-1)
    #weight to spin configuration
    c_s=weight_to_spin(c, c, L, nbit)
    return J_s, h_s, c_s

#linear layer
def linear_layer_to_ising(s, f1z, c, L, nbit=8):
    Q=s.shape[0]
    F=1/Q
    Nb=2**nbit-1
    #binary tensor 
    t_bin=torch.tensor([2**e for e in range(nbit)]).to(s.device)
    #J
    J_orig=F*contract('ai,aj->ij', s, s)
    J=(L/Nb)**2*contract('ij,q,w->iqjw', J_orig, t_bin, t_bin)
    J_s=J.flatten(2,-1).flatten(0,-2)
    #h
    h_orig=-2*F*contract('aj,ai->ij', s, f1z)
    h=h_orig+2*contract('jl,il->ij', J_orig, c)
    h=L/Nb*contract('ij,q->ijq', h, t_bin)
    h_s=h.flatten(1,-1)
    #weight to spin configuration
    c_s=weight_to_spin(c, c, L, nbit)
    return J_s, h_s, c_s
    
def layer_to_ising(s, f1z, c, k, L=1, nbit=8):
    #the weight c takes value in the range [-L,L]
    if len(s.shape)>2:
        return conv_layer_to_ising(s, f1z, c, k, L, nbit=nbit)
    else:
        return linear_layer_to_ising(s, f1z, c, L, nbit=nbit)
    
#conv layer act
def conv_layer_to_ising_act(s, gamma, f1z, c, K, L, nbit=8):
    Q=s.shape[0]
    N2=(s.shape[-1]-K+1)**2
    F=1/(Q*N2)
    Nb=2**nbit-1
    #unfold x for convolution (x_{c(i-k)(j-l)})
    s_patch= s.unfold(2, K, 1).unfold(3, K, 1)#.type(torch.float16)
    #activation factor (consider only training samples s_i whose output is >0)
    #s_patch = contract('abijkl,adij->abdijkl', s_patch, gamma)
    #binary tensor 
    t_bin=torch.tensor([2**e for e in range(nbit)]).to(s_patch.device)
    #J
    #J_orig=F*contract('abdijkl,acdijmn->dbklcmn', s_patch, s_patch)
    J_orig=F*contract('abijkl,acijmn,adij->dbklcmn', s_patch, s_patch, gamma**2)
    J=(L/Nb)**2*contract('dbklcmn,q,w->dbklqcmnw', J_orig, t_bin, t_bin)
    J_s=J.flatten(5,-1).flatten(1,-2)
    #h
    #h_orig=-2*F*contract('abcijkl,acij->bckl', s_patch, f1z)
    h_orig=-2*F*contract('abijkl,acij->bckl', s_patch, f1z*gamma)
    h=h_orig+2*contract('dbklcmn,dcmn->bdkl', J_orig, c)
    h=L/Nb*contract('bdkl,q->dbklq', h, t_bin)
    h_s=h.flatten(1,-1)
    #weight to spin configuration
    c_s=weight_to_spin(c, c, L, nbit)
    return J_s, h_s, c_s
    
#linear layer act
def linear_layer_to_ising_act(s, gamma, f1z, c, L, nbit=8):
    Q=s.shape[0]
    F=1./Q
    Nb=2**nbit-1
    #activation factor (consider only training samples s_i whose output is >0)
    s = contract('aj,ai->aji', s, gamma)
    #binary tensor
    t_bin=torch.tensor([2**e for e in range(nbit)]).to(s.device)
    #J
    J_orig=F*contract('aji,ali->ijl', s, s)
    J=(L/Nb)**2*contract('ijl,q,w->ijqlw', J_orig, t_bin, t_bin)
    J_s=J.flatten(3,-1).flatten(1,-2)
    #h
    h_orig=-2*F*contract('aji,ai->ij', s, f1z)
    h=h_orig+2*contract('ijl,il->ij', J_orig, c)
    h=L/Nb*contract('ij,q->ijq', h, t_bin)
    h_s=h.flatten(1,-1)
    #weight to spin configuration
    c_s=weight_to_spin(c, c, L, nbit)
    return J_s, h_s, c_s
        
def layer_to_ising_act(s, gamma, f1z, c, k, L=1, nbit=8):
    #the weight c takes value in the range [-L,L]
    if len(s.shape)>2:
        return conv_layer_to_ising_act(s, gamma, f1z, c, k, L, nbit=nbit)
    else:
        return linear_layer_to_ising_act(s, gamma, f1z, c, L, nbit=nbit)
    
#Ising hamiltonian
def H_Ising(J_s, h_s, c_s):
    return contract('qw,oq,ow->o', J_s, c_s.float(), c_s.float()) + contract('cq,cq->c', h_s, c_s.float())
    
def dH_Ising(J_s, h_s, c_s, c_s_new, i_c):
    J_s_c = J_s[i_c].clone()
    J_s_c[i_c]=0
    return 2*c_s_new*(2*torch.sum(J_s_c*c_s) + h_s[i_c])
    
#Multiprocessing Metropolis algorithm
def Metropolis_step(J_s, h_s, c_s, beta, h):
    J_s_h = J_s[h] if len(J_s.shape)>2 else J_s
    #cycle over each hamiltonian spin site     
    rnd_sites=torch.randperm(c_s.shape[1])
    for i in rnd_sites:
        #assign new random spin value to the site
        c_s_new=-c_s[h,i]
        #Calculate energy change due to new spin value
        dE = dH_Ising(J_s_h, h_s[h], c_s[h], c_s_new, i)
        #decide if site spin value should be updated (based on dE and beta)
        if dE <= 0 or torch.rand(1).to(dE.device) < torch.exp(-beta*dE):
            c_s[h,i] = c_s_new
            
#convert embedding json key from stored string to int
def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x
    
    
#Quantum annealing using dwave
def QuantumAnnealing(J_s, h_s, c_s, batch_idx, log_folder):
    path=f'{log_folder}/SG/embedding_{J_s.shape[0]}-{J_s.shape[1]}.json'
    conf_file = open(f"{log_folder}/configuration.json")
    conf = json.load(conf_file)
    conf_file.close()
    n_reads, t_ann = conf['n_reads'], conf['t_ann']
    #load stored embedding (if any)
    try:
        f = open(path,)
        embedding = json.load(f, object_hook=jsonKeys2int)
        f.close()
    except:
        embedding = None
    #create sampler
    if embedding == None:
        sampler = EmbeddingComposite(DWaveSampler())
    else:
        sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
    E_tot=0
    #use sampler to minimize each ising model
    for i in tqdm.tqdm(range(c_s.shape[0]), leave=False):
        h, J, c = h_s[i].numpy(), J_s.numpy(), c_s[i].numpy()
        if embedding == None:
            sampleset = sampler.sample_ising(h, J, return_embedding=True, 
                                             num_reads=n_reads, annealing_time=t_ann)
            embedding = sampleset.info["embedding_context"]["embedding"]
            sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
            with open(path, 'w') as f:
                json.dump(embedding, f)
        else:
            sampleset = sampler.sample_ising(h, J, num_reads=n_reads, annealing_time=t_ann)
        #return spin configuration of minimum energy
        c_s[i] = torch.tensor(list(sampleset.samples()[0].values()))
        E_tot+=list(sampleset.data(fields=['energy']))[0][0]
    return c_s, E_tot

##Quantum annealing using dwave with initial_state
#def QuantumAnnealing_in_state(J_s, h_s, c_s, log_folder):
#    path=f'{log_folder}/SG/embedding_{J_s.shape[0]}-{J_s.shape[1]}.json'
#    n_reads, t_ann = 500, 2
#    #load stored embedding (if any)
#    try:
#        f = open(path,)
#        embedding = json.load(f, object_hook=jsonKeys2int)
#    except:
#        embedding = None
#    #create sampler
#    if embedding == None:
#        sampler = EmbeddingComposite(DWaveSampler())
#    else:
#        sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
#    E_tot=0
#    #use sampler to minimize each ising model
#    for i in tqdm.tqdm(range(c_s.shape[0]), leave=False):
#        h, J, c = h_s[i].numpy(), J_s.numpy(), c_s[i].numpy()
#        h_q = {k:h[k] for k in range(len(h))}
#        J_q = {(k,l):J[k,l] for k in range(J.shape[0]) for l in range(J.shape[1])}
#        c_q = {k:c[k] for k in range(len(c))}
#        schedule = [[0.0, 1.0], [t_ann/2, 0.45], [t_ann, 1.0]]
#        if embedding == None:
#            sampleset = sampler.sample_ising(h_q, J_q, initial_state=c_q, return_embedding=True, 
#                                             num_reads=n_reads, anneal_schedule=schedule,
#                                             reinitialize_state=True, answer_mode='histogram')
#            embedding = sampleset.info["embedding_context"]["embedding"]
#            sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
#            with open(path, 'w') as f:
#                json.dump(embedding, f)
#        else:
#            sampleset = sampler.sample_ising(h_q, J_q, initial_state=c_q, num_reads=n_reads, 
#                                             anneal_schedule=schedule, answer_mode='histogram',
#                                             reinitialize_state=True)
#        #return spin configuration of minimum energy
#        c_s[i] = torch.tensor(list(sampleset.samples()[0].values()))
#        E_tot+=list(sampleset.data(fields=['energy']))[0][0]
#    return c_s, E_tot

        
#############################
#FFT deconvolution layer
class Deconv2dfft(torch.nn.Module):
    def __init__(self, c2, c1, kernel_size, image_size):
        super().__init__()
        self.w_k = kernel_size
        self.w_img = image_size
        wk2 = self.w_k//2
        w = image_size + 2*wk2 - 1
        self.weight=nn.parameter.Parameter(torch.zeros((c2,c1,self.w_k+w,self.w_k+w)))

    def forward(self, x):
        k_fft_inv = self.weight
        w_k = h_k = self.w_k
        wk2, hk2 = w_k//2, h_k//2
        #image and kernel must both be of size (w_img+w_ker-1)x(h_img+h_ker-1)
        x=F.pad(x, (wk2, wk2+w_k-1, hk2, hk2+h_k-1))
        #compute image fft
        image_fft=torch.fft.fft2(x)
        #image_fft = torch.rfft(x, 2, normalized=True, onesided=False)
        deconv_fft = contract('ackl,jckl->ajkl', image_fft, k_fft_inv)
        #compute inverse fft
        deconv_c1=torch.fft.ifft2(deconv_fft)
        #crop result 
        deconv_c1=deconv_c1[:,:,wk2:-wk2,wk2:-wk2]
        return deconv_c1.real        
        
    #FFT inverse convolution kernel
    def inverse(self, kernel_c2c1):
        C2, C1 = kernel_c2c1.shape[:2]
        w_k, h_k = 2*(kernel_c2c1.shape[2]//2), 2*(kernel_c2c1.shape[3]//2)
        w, h = self.w_img + w_k, self.w_img + h_k
        #image and kernel must both be of size (w_img+w_ker-1)x(h_img+h_ker-1)
        kernel_c2c1=F.pad(kernel_c2c1, (0, w-1, 0, h-1), "constant", 0)
        #compute kernel fft
        k_fft=torch.fft.fft2(kernel_c2c1)
        #k_fft=torch.rfft(kernel_c2c1, 2, normalized=True, onesided=False)
        #transpose kernel fft array (c2, c1, w, h) --> (w, h, c2, c1)
        k_fft_t=torch.permute(k_fft,(2,3,0,1))
        #compute kernel fft inverse matrix on indices (c2, c1) (for each pixel)  
        k_fft_t_inv=torch.linalg.pinv(k_fft_t)
        #transpose inverse kernel fft array back (w, h, c1, c2) --> (c1, c2, w, h)
        k_fft_inv=torch.permute(k_fft_t_inv,(2,3,0,1))   
        self.weight = nn.Parameter(torch.conj(k_fft_inv), requires_grad=False)
        #return nn.Parameter(torch.conj(k_fft_inv), requires_grad=False)


#############################
#Spectral pooling layer
class SpectralPool2d(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale = nn.modules.utils._pair(scale_factor)

    #DHT transform
    def DiscreteHartleyTransform(self, input):
        fft = torch.fft.fft2(input, norm="forward")
        fft = torch.fft.fftshift(fft)
        return fft.real - fft.imag
    #Inverse DHT transform
    def InverseDiscreteHartleyTransform(self, input):
        dht = torch.fft.ifftshift(input)
        fft = torch.fft.fft2(dht, norm="backward")
        return fft.real - fft.imag

    def forward(self, input):
        dht = self.DiscreteHartleyTransform(input)
        W,H=dht.shape[-2],dht.shape[-1]
        w,h = int(W*self.scale[0]), int(H*self.scale[1])
        dx, dy = abs(W-w)//2, abs(H-h)//2
        rx, ry = abs(W-w)%2, abs(H-h)%2
        if self.scale[0] < 1:
            #pool
            dht = dht[:,:,dx:-dx-rx,dy:-dy-rx]
        else:
            #unpool
            dht=torch.nn.functional.pad(dht, (dy, dy+ry, dx, dx+rx))
        return self.InverseDiscreteHartleyTransform(dht)
