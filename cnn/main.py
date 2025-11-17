import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torchmetrics
import logging
import libannealing as la
import libnetworks as ln
import libcnn as lc
import torch.multiprocessing as mp
import tqdm
from itertools import repeat
import copy
import signal
import json
import time

#disable tracebackon ctrl+c
signal.signal(signal.SIGINT, lambda x, y: sys.exit())
#disable logging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
#do not use gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#disable scientific notation
torch.set_printoptions(sci_mode=False, threshold=20000)

#################################
#configuration

#network
networks={0: (ln.single_layer, ln.single_layer_bw), 
          1: (ln.single_layer_small, ln.single_layer_small_bw), 
          2: (ln.single_layer_very_small, ln.single_layer_very_small_bw), 
          3: (ln.two_layer, ln.two_layer_bw), 
          4: (ln.two_layer_very_small, ln.two_layer_very_small_bw), 
          5: (ln.three_layer, ln.three_layer_bw), 
          6: (ln.linear, ln.linear_bw), 
          7: (ln.single_layer_conv, ln.single_layer_conv_bw), 
          8: (ln.single_layer_conv_small, ln.single_layer_conv_small_bw), 
          9: (ln.two_layer_conv, ln.two_layer_conv_bw), 
         10: (ln.two_layer_conv_small, ln.two_layer_conv_small_bw), 
         11: (ln.three_layer_conv, ln.three_layer_conv_bw),
         12: (ln.cnn_no_pool, ln.cnn_no_pool_bw), 
         13: (ln.lenet_small, ln.lenet_small_bw), 
         14: (ln.lenet, ln.lenet_bw)}
          
net, net_backward = networks[14]

#dataset
datasets=[MNIST, FashionMNIST, CIFAR10]
DATASET=datasets[0]
#get nr of input channels
in_channel = 3 if DATASET == CIFAR10 else 1

#training approach
trainings=["backpropagation", "simulated annealing", "quantum annealing", "simulated quantum annealing", "imaginary time evolution"]
training=trainings[1]

#training configuration
N_epochs=10 #training epochs
N_samples=1000 #number of samples to use for training 
bsize=1 #batch size (used by backpropagation)
alpha=0.1 #LeakyReLU negative slope





#annealing parameters from network configuration file
conf=None
with open(f"./saved_models/{net.__name__}/configuration.json") as conf_file:
    conf = json.load(conf_file)
nbit=conf['nbit'] #number of bits in binary representation
n_rep=conf['n_rep'] #number of times Metropolis algorithm is run at each step 
beta_in=conf['beta_in'] # intial beta (inverse temperature)
beta_final=conf['beta_final'] # final beta
L_list=conf['L_list'] #range of the weights in each layer (around current value)
n_reads=conf['n_reads'] #Number of reads in quantum annealing
t_ann=conf['t_ann'] #Annealing time in quantum annealing 
n_shots=conf['n_shots'] #Number of measurements in ITEMC
tau=conf['tau'] #Imaginary time to evolve the system in ITEMC 
layer_list=conf['layer_list'] #indices of layers to be trained

#################################
#paths
log_folder=f"./saved_models/{net.__name__}/"
os.makedirs(log_folder, exist_ok=True)
os.makedirs(log_folder+'SG/', exist_ok=True)
os.makedirs(log_folder+'BP/', exist_ok=True)

#################################
#plot current saved data
if len(sys.argv)>1 and sys.argv[1] == 'plot':
    lc.plot_acc(log_folder)
    exit()

#################################
#init network
#torch.manual_seed(42)
cnn = lc.CNN(n_cat = 10, in_channel = in_channel, img_width = 32, alpha = alpha, 
          train_list = layer_list, network=net, network_backward=net_backward)

#################################
#Dataset
transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
dataset = DATASET(root="./data", train=True, download=True, transform=transform)

#split to train and validation datasets
N_train = int(len(dataset)*(1-0.1))
N_val = len(dataset) - N_train
train_set, val_set = data.random_split(dataset, [N_train, N_val])
#resize training set
N_samples=min(N_samples,len(train_set))
indices = torch.arange(N_samples)
train_set = torch.utils.data.Subset(train_set, indices)
#for annealing use a single batch
bsize = bsize if training == "backpropagation" else len(train_set)
#train and validation dataloaders
train_loader = data.DataLoader(train_set, batch_size=bsize, num_workers=4)
val_loader = data.DataLoader(val_set, batch_size=len(val_set), num_workers=4)
#get test dataset
dataset_test = DATASET(root="./data", train=False, download=True, transform=transform)
test_loader = data.DataLoader(dataset_test, batch_size=len(dataset_test), num_workers=4)

#################################
#summary
lc.summary(cnn, train_set[0][0].shape)
print(f"Train by: {training}")
print(f"N samples: {N_samples}")
print(f"Batch size: {bsize}")
print(f"N epochs: {N_epochs}")
if training != "backpropagation":
    print(f"N bits: {nbit}")
    if training == "simulated annealing":
        print(f"Beta initial: {beta_in}")
        print(f"Beta final: {beta_final}")
    elif training.endswith("quantum annealing"):
        print(f"N reads: {n_reads}")
        print(f"Annealing time: {t_ann}")
    elif training == "imaginary time evolution":
        print(f"N shots: {n_shots}")
        print(f"imaginary time: {tau}")

#################################
#Train
metrics = lc.MetricTracker()
accelerator = 'gpu' if training=="backpropagation" else 'cpu'
trainer = pl.Trainer(default_root_dir=log_folder, max_epochs=N_epochs, accelerator=accelerator, logger=False, enable_checkpointing=False, callbacks=[EarlyStopping(monitor="val_acc", mode="max", patience=20), metrics], enable_progress_bar=training=="backpropagation")

# train the model by backpropagation
if training=="backpropagation":
    cnn.train()
    cnn.cuda()
    #cnn.cpu()
    t0=time.time()
    trainer.fit(model=cnn, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(log_folder+"BP/checkpoint.ckpt")
    print("training time:",time.time()-t0)
    # Test model on validation and test set
    test_result = trainer.test(cnn, test_loader, verbose=False)
    print("Test results:")
    print(test_result)
    #plot accuracy 
    lc.save_plot_acc(metrics.collection, log_folder, label=0)
    
#Train model by simulated/quantum annealing
else:
    training_step = cnn.training_step_ising if alpha==1 else cnn.training_step_ising_act
    label = 1
    if training == "quantum annealing":
        label = 2
    elif training == "simulated quantum annealing":
        label = 3
    elif training.endswith("evolution"):
        label = 4
    with torch.no_grad():
        cnn.cpu()
        train_iterator=iter(train_loader)
        val_iterator=iter(val_loader)
        x_val, y_val = next(val_iterator)
        N_batches=len(train_iterator)
        # Test model on validation and test set
        test_result = trainer.test(cnn, test_loader, verbose=False)
        max_acc=test_result[0]['test_acc']
        acc=test_result[0]['test_acc']
        k_max_acc=0
        list_acc=[acc]
        t0=time.time()
        for k in range(N_epochs):
            #train over batches
            for i in range(N_batches):
                x, y = next(train_iterator)
                training_step([x.to(cnn.device), y.to(cnn.device)], k, N_epochs, 
                                        acc, nbit=nbit, n_rep=n_rep, beta_in=beta_in, 
                                        beta_final=beta_final, L_list=L_list, 
                                        layer_list=layer_list,
                                        alpha=alpha, training=training, log_folder=log_folder, 
                                        n_reads=n_reads, t_ann=t_ann, n_shots=n_shots, tau=tau)
            # Test model on validation and test set
            test_result = trainer.test(cnn, val_loader, verbose=False)
            acc=test_result[0]['test_acc']
            #save best weights
            list_acc.append(acc)
            if acc>max_acc:
                max_acc, k_max_acc = acc, k
                torch.save(cnn, log_folder+'SG/model.pt')
            train_iterator=iter(train_loader)   
        print("training time:",time.time()-t0)         
    print("max_acc", max_acc, "at step", k_max_acc+1)
    #load max accuracy weights
    cnn = torch.load(log_folder+'SG/model.pt', weights_only=False)
    # Test model on validation and test set
    test_result = trainer.test(cnn, test_loader, verbose=False)
    print("test results")
    print(test_result)
    #plot accuracy 
    lc.save_plot_acc(list_acc, log_folder, label=label)
    
    
##backward -> forward accuracy
#y=torch.tensor([val_set[i][1] for i in range(1000)])
#x=cnn.network_backward.forward_return_all(y)[-1]
#y1=cnn(x)
#y2=torch.tensor([torch.argmax(y1[i]).item() for i in range(len(y1))])
#good = len(y) - len((y-y2).nonzero())
#print(good)
#exit()



