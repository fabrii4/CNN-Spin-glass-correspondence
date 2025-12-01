import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torchmetrics
import logging
import libannealing as la
import torch.multiprocessing as mp
import tqdm
from itertools import repeat
import signal
from termcolor import colored
import matplotlib.pyplot as plt


#disable tracebackon ctrl+c
signal.signal(signal.SIGINT, lambda x, y: sys.exit())

#disable logging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

#do not use gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#disable scientific notation
torch.set_printoptions(sci_mode=False, threshold=20000)

##################################
# define the network LightningModule
class CNN(pl.LightningModule):
    def __init__(self, n_cat, in_channel, img_width, alpha, train_list, network, network_backward):
        super().__init__()
        self.network = network(n_cat, in_channel, alpha, train_list)
        self.network_backward = network_backward(n_cat, img_width, alpha)
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass",num_classes=n_cat)
        self.init()
        
    def _loss(self, batch):
        x, y = batch
        z = self.network(x)
        loss = F.mse_loss(z, F.one_hot(y, num_classes=10)*1.)
        #loss = F.cross_entropy(z, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.network(x)
        loss = self._loss(batch)
        acc = self.accuracy(preds, y)
        self.log("train_loss", loss)
        self.log('train_acc', acc)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.network(x)
        loss = self._loss(batch)
        acc = self.accuracy(preds, y)
        self.log("val_loss", loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.network(x)
        loss = self._loss(batch)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
        return [optimizer], [scheduler]
        
    def forward(self, x):
        z = self.network(x)
        return z
        
    def update(self):
        self.network_backward.update(self.network)

    def init(self):
        self.network.init()
        self.update()
        
    def training_step_ising(self, batch, batch_idx, N_steps, acc, nbit, n_rep, beta_in, beta_final, L_list,  layer_list, alpha, training, log_folder, n_reads, t_ann, n_shots, tau):
        #beta
        g=np.log(beta_final/beta_in)/(N_steps)
        beta=beta_in*np.exp(g*batch_idx)
        #propagate training batch backward
        x, y = batch
        preds_back = self.network_backward.forward_return_all(y)
        #cycle over layers
        layer_list = np.array(layer_list)
        for i_layer in layer_list:
            #layer weights range
            L_i, L_f = L_list[i_layer]
            g=np.log(L_f/L_i)/(N_steps)
            L=L_i*np.exp(g*batch_idx)
            #propagate training batch forward
            preds = self.network.forward_return_all(x)
            #extract input, output and weights of each layer
            s=preds[i_layer]
            f1z=preds_back[len(self.network.layer_list)-i_layer-1]
            c=self.network.layer_list[i_layer].weight
            #transform layer to ising configuration
            k_size = 0 if len(s.shape)<=2 else self.network.layer_list[i_layer].kernel_size[0]
            J_s, h_s, c_s = la.layer_to_ising(s, f1z, c, k_size, L, nbit)
            #minimize hamiltonians
            if training=="quantum annealing":
                c_s, E_tot = la.QuantumAnnealing(J_s, h_s, c_s, batch_idx, log_folder)
                #c_s, E_tot = la.QuantumAnnealing_in_state(J_s, h_s, c_s, log_folder)
                print("step "+str(batch_idx+1)+"/"+str(N_steps), "layer", i_layer, "acc "+"{:.4f}".format(acc), "E_tot", E_tot)
            elif training=="simulated quantum annealing":
                #multiprocess with pool
                n_proc=min(mp.cpu_count(),c_s.shape[0])
                args=zip(repeat(J_s), repeat(h_s), repeat(c_s), repeat(n_reads), 
                         repeat(t_ann), range(c_s.shape[0]))
                with mp.Pool(n_proc, initializer=torch.seed) as pool:
                    pool.starmap(la.SimulatedQuantumAnnealing, args)
                print("step "+str(batch_idx+1)+"/"+str(N_steps), "layer", i_layer, "acc "+"{:.4f}".format(acc))#, "E_tot", E_tot)
            elif training =="imaginary time evolution":
                #multiprocess with pool
                n_proc=min(mp.cpu_count(),c_s.shape[0])
                args=zip(repeat(J_s), repeat(h_s), repeat(c_s), repeat(n_shots), 
                         repeat(tau), range(c_s.shape[0]))
                with mp.Pool(n_proc, initializer=torch.seed) as pool:
                    pool.starmap(la.ITEMC, args)
                print("step "+str(batch_idx+1)+"/"+str(N_steps), "layer", i_layer, "acc "+"{:.4f}".format(acc))
            else:
                #classical
                info="step "+str(batch_idx+1)+"/"+str(N_steps)+" layer "+str(i_layer)+" acc "+"{:.4f}".format(acc)
                for i in tqdm.tqdm(range(n_rep), leave=False, desc=info):
                    #multiprocess with pool
                    n_proc=min(mp.cpu_count(),c_s.shape[0])
                    args=zip(repeat(J_s), repeat(h_s), repeat(c_s), 
                             repeat(beta), range(c_s.shape[0]))
                    with mp.Pool(n_proc, initializer=torch.seed) as pool:
                        pool.starmap(la.Metropolis_step, args)
            #update layer weights from spin configuration
            self.network.layer_list[i_layer].weight=la.spin_to_weight(c_s, c, L, nbit)
        #update weights in backward network
        self.update()
        
    def training_step_ising_act(self, batch, batch_idx, N_steps, acc, nbit, n_rep, beta_in, beta_final, L_list,  layer_list, alpha, training, log_folder, n_reads, t_ann, n_shots, tau):
        n_rep = n_rep if batch_idx<10 else 1
        #beta
        g=np.log(beta_final/beta_in)/(N_steps)
        beta=beta_in*np.exp(g*batch_idx)
        #propagate training batch backward
        x, y = batch
        preds_back = self.network_backward.forward_return_all(y)
        #cycle over layers
        layer_list = np.array(layer_list)
        for i_layer in layer_list:
            #layer weights range
            L_i, L_f = L_list[i_layer]
            g=np.log(L_f/L_i)/(N_steps)
            L=L_i*np.exp(g*batch_idx)
            #propagate training batch forward
            preds = self.network.forward_return_all(x)
            #activation factor
            gamma = preds[len(preds)//2+1+i_layer].sign() #non-activated layers outputs
            gamma[gamma<0]=alpha
            #f1z*=gamma
            #activation factors for all layers
            #s1 = preds[len(preds)//2+1:] #non-activated layers outputs
            #gamma=[s1[i].sign() for i in layer_list]#[layer_list>=i_layer]]
            #l_alpha = [alpha for i in layer_list]
            #l_alpha = [l_alpha[i] for i in layer_list[layer_list>=i_layer]]
            #for i in range(len(gamma)): gamma[i][gamma[i]<0]=l_alpha[i]
            #take into account activation factors of higher layers
            #b=torch.ones(gamma[-1].shape)
            #for i in range(len(gamma)-1):
            #    w=self.network.layer_list[-i-1].weight
            #    b=torch.einsum('gi,ag,ag->ai',w,gamma[-i-1],b)
            #f1z*=b
            #gamma=gamma[0]#*b
            #propagate training batch backward
            #preds_back = self.network_backward.forward_return_all(y, gamma)
            #extract input, output and weights of each layer
            s=preds[i_layer]
            f1z=preds_back[len(self.network.layer_list)-i_layer-1]
            c=self.network.layer_list[i_layer].weight
            #transform layer to ising configuration
            k_size = 0 if len(s.shape)<=2 else self.network.layer_list[i_layer].kernel_size[0]
            J_s, h_s, c_s = la.layer_to_ising_act(s, gamma, f1z, c, k_size, L, nbit)
            #minimize hamiltonians
            if training=="quantum annealing":
                #quantum
                c_s, E_tot = la.QuantumAnnealing(J_s, h_s, c_s, batch_idx, log_folder)
                #c_s, E_tot = la.QuantumAnnealing_in_state(J_s, h_s, c_s, log_folder)
                print("step "+str(batch_idx+1)+"/"+str(N_steps), "layer", i_layer, "acc "+"{:.4f}".format(acc), "E_tot", E_tot)
            elif training=="simulated quantum annealing":
                #quantum
                #c_s, E_tot = la.SimulatedQuantumAnnealing(J_s, h_s, c_s, batch_idx, log_folder)
                #c_s, E_tot = la.QuantumAnnealing_in_state(J_s, h_s, c_s, log_folder)
                #multiprocess with pool
                n_proc=min(mp.cpu_count(),c_s.shape[0])
                args=zip(repeat(J_s), repeat(h_s), repeat(c_s), repeat(n_reads), 
                         repeat(t_ann), range(c_s.shape[0]))
                with mp.Pool(n_proc, initializer=torch.seed) as pool:
                    pool.starmap(la.SimulatedQuantumAnnealing, args)
                print("step "+str(batch_idx+1)+"/"+str(N_steps), "layer", i_layer, "acc "+"{:.4f}".format(acc))#, "E_tot", E_tot)
            elif training =="imaginary time evolution":
                #multiprocess with pool
                n_proc=min(mp.cpu_count(),c_s.shape[0])
                args=zip(repeat(J_s), repeat(h_s), repeat(c_s), repeat(n_shots), 
                         repeat(tau), range(c_s.shape[0]))
                with mp.Pool(n_proc, initializer=torch.seed) as pool:
                    pool.starmap(la.ITEMC, args)
                print("step "+str(batch_idx+1)+"/"+str(N_steps), "layer", i_layer, "acc "+"{:.4f}".format(acc))    
            else:
                #classical
                info="step "+str(batch_idx+1)+"/"+str(N_steps)+" layer "+str(i_layer)+" acc "+"{:.4f}".format(acc)
                for i in tqdm.tqdm(range(n_rep), leave=False, desc=info):
                    #multiprocess with pool
                    n_proc=min(mp.cpu_count(),c_s.shape[0])
                    args=zip(repeat(J_s), repeat(h_s), repeat(c_s), 
                             repeat(beta), range(c_s.shape[0]))
                    with mp.Pool(n_proc, initializer=torch.seed) as pool:
                        pool.starmap(la.Metropolis_step, args)
            #update layer weights from spin configuration
            self.network.layer_list[i_layer].weight=la.spin_to_weight(c_s, c, L, nbit)
        #update weights in backward network
        self.update()

##################################
#summary
def summary(cnn, input_shape):
    net_name=str(cnn.network).split("(")[0]
    print(f"Network \033[1m{net_name}\033[0m summary:")
    print("--------------------------")
    modules, handles = [], []
    def add_hook(m):
        def forward_hook(module, input, output):
            modules.append(module)
        handle=m.register_forward_hook(forward_hook)
        handles.append(handle)
    cnn.network.apply(add_hook)
    input = torch.rand(1,*input_shape)
    cnn.network.forward(input)  # hooks are fired sequentially from model input to the output
    for i in range(len(modules)):
        if hasattr(modules[i], 'weight'):
            if modules[i].weight.requires_grad:
                print(colored(f"{i} {modules[i]} train: True", "green"))
            else:
                print(colored(f"{i} {modules[i]} train: False", "red"))
        else:
            print(f"{i} {modules[i]}")
    print("--------------------------")
    for handle in handles:
        handle.remove()
        
##################################
#Metrics callback and plotting 
class MetricTracker(Callback):
  def __init__(self):
    self.collection = []
  def on_validation_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics['val_acc'].item() # access it here
    self.collection.append(elogs)
    # do whatever is needed

##################################    
#plot accuracy 
def plot_acc(path):
    #load accuracy log
    acc_log = np.genfromtxt(path+'acc_log.csv', delimiter=',')
    if acc_log.ndim == 1:
        acc_log=np.array([acc_log])
    #keep only initial 2 and last 3 results
    acc_log=np.delete(acc_log,slice(3,-3), axis=0)
    #plot style
    color = plt.cm.rainbow(np.linspace(0, 1, len(acc_log)))
    color = np.insert(color,0, [1,0,0,1], axis=0)
    color=np.flip(color[:len(acc_log)],axis=0)
    labels=['1st','2nd','3nd','3rd last', '2nd last','last',]
    type_l=[' BP', ' SA', ' QA', ' SQA', ' ITE']
    #plot
    plt.style.use('dark_background')
    plt.figure(figsize=(15,10))
    for i in range(len(acc_log)):
        acc, typ, c = acc_log[i][:-1], type_l[int(acc_log[i][-1])], color[i]
        plt.plot(acc, ['-o','--s'][i>2], c=c, label=labels[i]+typ, linewidth=2, markersize=10)
        plt.axhline(y=max(acc), linestyle='--', c=c, linewidth=1)
    #plt.ylim(0, 1)
    plt.legend(loc='lower right', prop={'size': 20})
    plt.savefig(path+'acc_graph.png')
    #plt.savefig(path+'acc_graph.eps', format='eps')
    plt.show()
    
def save_plot_acc(list_acc, path, label):
    L=21
    list_acc=np.array([list_acc])
    #save accuracy log
    l=len(list_acc[0])
    if l > 1:
        if l < L:
            list_acc = np.pad(list_acc, ((0,0), (0,L-l)), constant_values=list_acc[0,-1])
        elif l > L:
            idx = np.round(np.linspace(0,l-1, L)).astype(int)
            list_acc = list_acc[:,idx]
        with open(path+'acc_log.csv', "ab") as f:
            np.savetxt(f, np.append(list_acc,[[label]],axis=1), delimiter=",")
    #plot accuracy
    plot_acc(path)

