import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import make_grid
import pytorch_lightning as pl
import tqdm
import json
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite





###################################
#convert to binary representation
def dec2bin(x, bits=8):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    
def bin2dec(b, bits=8):
    mask = 2 ** torch.arange(bits).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

###################################
#Neural to spin representation conversion
def conv_layer_to_ising(s, f1z, c, K, stride, L, nbit=8):
    Q=s.shape[0]
    N2=(s.shape[-1]-K+1)**2
    F=1/(Q*N2)
    #F=1/Q
    Nb=(2**nbit-1)
    #unfold x for convolution (x_{c(i-k)(j-l)})
    s_patch= s.unfold(2, K, stride).unfold(3, K, stride)
    #binary tensor 
    t_bin=torch.tensor([2**e for e in range(nbit)]).to(s_patch.device)
    #J
    J_orig=F*torch.einsum('abijkl,acijmn->bklcmn', s_patch, s_patch)
    J=4*(L/Nb)**2*torch.einsum('bklcmn,q,w->bklqcmnw', J_orig, t_bin, t_bin)
    J=J.flatten(4,-1).flatten(0,-2)
    J_s=J/4
    #h
    h_orig=-2*F*torch.einsum('abijkl,acij->bckl', s_patch, f1z)
    h_J=torch.einsum('cklbmn->ckl', J_orig).unsqueeze(1).repeat(1,h_orig.shape[1],1,1)
    h=2*L*h_orig-4*L**2*h_J
    h=1/Nb*torch.einsum('bckl,q->cbklq', h, t_bin)
    h=h.flatten(1,-1)
    h_s=h/2+J.sum(dim=-1)/2
    #E
    #E_orig=F*torch.einsum('acij,acij->c', f1z, f1z)
    #E=E_orig+L**2*J_orig.sum()-L*torch.einsum('bckl->c', h_orig)
    #E_s=E+h.sum(dim=-1)/2+J.sum()/4
    E_s=0
    #weight to binary configuration
    c_orig=c
    c=(((c+L)/(2*L))*Nb).round().long()
    c=dec2bin(c, bits=nbit)
    c=c.flatten(1,-1)
    c_s=2*c-1
    #return J_s.triu(), h_s, E_s, c_s
    return J_s, h_s, E_s, c_s
            
def linear_layer_to_ising(s, f1z, c, L, nbit=8):
    Q=s.shape[0]
    F=1/Q
    #F=1
    Nb=(2**nbit-1)
    #binary tensor 
    t_bin=torch.tensor([2**e for e in range(nbit)]).to(s.device)
    #J
    J_orig=F*torch.einsum('ai,aj->ij', s, s)
    J=4*(L/Nb)**2*torch.einsum('ij,q,w->iqjw', J_orig, t_bin, t_bin)
    J=J.flatten(2,-1).flatten(0,-2)
    J_s=J/4
    #h
    h_orig=-2*F*torch.einsum('aj,ai->ij', s, f1z)
    h_J=torch.einsum('ij->j', J_orig).unsqueeze(0).repeat(h_orig.shape[0],1)
    h=2*L*h_orig-4*L**2*h_J
    h=1/Nb*torch.einsum('ij,q->ijq', h, t_bin)
    h=h.flatten(1,-1)
    h_s=h/2+J.sum(dim=-1)/2
    #E
    #E_orig=F*torch.einsum('ai,ai->i', f1z, f1z)
    #E=E_orig+L**2*J_orig.sum()-L*torch.einsum('ij->i', h_orig)
    #E_s=E+h.sum(dim=-1)/2+J.sum()/4
    E_s=0
    #weight to binary configuration
    c_orig=c
    c=(((c+L)/(2*L))*Nb).round().long()
    c=dec2bin(c, bits=nbit)
    c=c.flatten(1,-1)
    c_s=2*c.long()-1
    #return J_s.triu(), h_s, E_s, c_s
    return J_s, h_s, E_s, c_s
    
def layer_to_ising(s, f1z, c, k, stride, L=1, nbit=8):
    #the weight c takes value in the range [-L,L]
    if len(s.shape)>2:
        return conv_layer_to_ising(s, f1z, c, k, stride, L, nbit=nbit)
    else:
        return linear_layer_to_ising(s, f1z, c, L, nbit=nbit)

#spin configuration to weight conversion            
def spin_to_weight(c_s, c_shape, L=1, nbit=8):
    #the weight c takes value in the range [-L,L]
    Nb=(2**nbit-1)
    shape=torch.cat((torch.tensor(c_shape),torch.tensor([nbit])))
    c=(c_s+1)/2
    c=c.reshape(*shape)
    c=bin2dec(c, bits=nbit)
    c=L*(2*c/Nb-1)
    return nn.Parameter(c)
    
#Ising hamiltonian
def H_Ising(J_s, h_s, E_s, c_s):
    return torch.einsum('qw,oq,ow->o', J_s, c_s.float(), c_s.float()) + torch.einsum('cq,cq->c', h_s, c_s.float()) + E_s
    
def dH_Ising(J_s, h_s, c_s, c_s_new, i_h, i_c):
    J_s[i_c, i_c]=0
    return (c_s_new - c_s[i_h,i_c])*(2*torch.sum(J_s[i_c]*c_s[i_h]) + h_s[i_h,i_c])

    
#Multiprocessing Metropolis algorithm
def Metropolis_step(J_s, h_s, c_s, beta, h):
    #cycle over each hamiltonian spin site 
    rnd_sites=torch.randperm(c_s.shape[1])
    for i in rnd_sites:
        #assign new random spin value to the site
        c_s_new=-c_s[h,i]
        #Calculate energy change due to new spin value
        dE = dH_Ising(J_s, h_s, c_s, c_s_new, h, i)
        #decide if site spin value should be updated (based on dE and beta)
        if dE <= 0 or torch.rand(1).to(dE.device) < torch.exp(-beta*dE):
            c_s[h,i] = c_s_new
            
#convert embedding json key from stored string to int
def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

#Hybrid quantum annealing using dwave
def QuantumAnnealing(J_s, h_s, c_s):
    path=f'./saved_models/lenet/SG/embedding_{J_s.shape[0]}-{J_s.shape[1]}.json'
    n_reads, t_ann = 100, 2
    #load stored embedding (if any)
    try:
        f = open(path,)
        embedding = json.load(f, object_hook=jsonKeys2int)
    except:
        embedding = None
    #create sampler
    if embedding == None:
        sampler = EmbeddingComposite(DWaveSampler())
    else:
        sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
    E_tot=0
    #use sampler to minimize each ising model
    for i in tqdm.tqdm(range(c_s.shape[0])):
        h, J, c = h_s[i].numpy(), J_s.numpy(), c_s[i].numpy()
        h = {i:h[i] for i in range(len(h))}
        J = {(i,j):J[i,j] for i in range(J.shape[0]) for j in range(J.shape[1])}
        c = {i:c[i] for i in range(len(c))}
        if embedding == None:
            sampleset = sampler.sample_ising(h, J, initial_state=c, return_embedding=True, 
                                             num_reads=n_reads, annealing_time=t_ann,
                                             answer_mode='histogram')
            embedding = sampleset.info["embedding_context"]["embedding"]
            sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
            with open(path, 'w') as f:
                json.dump(embedding, f)
        else:
            sampleset = sampler.sample_ising(h, J, initial_state=c, num_reads=n_reads, 
                                             annealing_time=t_ann, answer_mode='histogram')
        #return spin configuration of minimum energy
        c_s[i] = torch.tensor(list(sampleset.samples()[0].values()))
        E_tot+=list(sampleset.data(fields=['energy']))[0][0]
    return c_s, E_tot


###################################
#Activation functions

#activation function Exp
class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.exp(input)
        
# inverse activation function Exp
class Log(nn.Module):
    def __init__(self, delta=0):
        super().__init__()
        self.delta=delta

    def forward(self, input):
        return torch.log(input+self.delta)
        
# inverse activation function Tanh
class Atanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.atanh(input)
        
# inverse activation function LeakyReLU
class LeakyReLUInv(nn.Module):
    def __init__(self, negative_slope = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        return input if input >= 0 else input/self.negative_slope
        
#############################
#FFT deconvolution layer
class Deconv2dfft(torch.nn.Module):
    def __init__(self, c2, c1, kernel_size, image_size):
        super().__init__()
        self.w_k = kernel_size
        wk2 = int((self.w_k-1)/2)
        w = image_size + 2*wk2 - 1
        self.weight=nn.parameter.Parameter(torch.zeros((c2,c1,self.w_k+w,self.w_k+w)))

    def forward(self, x):
        k_fft_inv = self.weight
        w_k = h_k = self.w_k
        wk2, hk2 = int((w_k-1)/2), int((h_k-1)/2)
        #image and kernel must both be of size (w_img+w_ker-1)x(h_img+h_ker-1)
        x=F.pad(x, (wk2, wk2+w_k-1, hk2, hk2+h_k-1))
        #compute image fft
        image_fft=torch.fft.fft2(x)
        deconv_fft = torch.einsum('inkl,jnkl->ijkl', image_fft, k_fft_inv)
        #compute inverse fft
        deconv_c1=torch.fft.ifft2(deconv_fft)
        #crop result 
        deconv_c1=deconv_c1[:,:,wk2:-wk2,wk2:-wk2]
        return deconv_c1        
        
#FFT inverse convolution kernel
def inverse_kernel_fft(kernel_c2c1, w_img):
    C2, C1 = kernel_c2c1.shape[:2]
    w_k, h_k = kernel_c2c1.shape[2:]
    wk2, hk2 = int((w_k-1)/2), int((h_k-1)/2)
    w, h = w_img + 2*wk2, w_img + 2*hk2
    #image and kernel must both be of size (w_img+w_ker-1)x(h_img+h_ker-1)
    kernel_c2c1=F.pad(kernel_c2c1, (0, w-1, 0, h-1), "constant", 0)
    #compute kernel fft
    k_fft=torch.fft.fft2(kernel_c2c1)
    #transpose kernel fft array (c2, c1, w, h) --> (w, h, c2, c1)
    k_fft_t=torch.permute(k_fft,(2,3,0,1))
    #compute kernel fft inverse matrix on indices (c2, c1) (for each pixel)  
    k_fft_t_inv=torch.linalg.pinv(k_fft_t)
    #transpose inverse kernel fft array back (w, h, c1, c2) --> (c1, c2, w, h)
    k_fft_inv=torch.permute(k_fft_t_inv,(2,3,0,1))   
    return nn.Parameter(torch.conj(k_fft_inv), requires_grad=False)

