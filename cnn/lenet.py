import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torchmetrics
import matplotlib.pyplot as plt
import logging
import liblenet as ll
import torch.multiprocessing as mp
import tqdm
from itertools import repeat
import copy

#disable logging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

#do not use gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#disable scientific notation
torch.set_printoptions(sci_mode=False, threshold=20000)


#############################
#network

class Lenet(pl.LightningModule):
    def __init__(self, n_cat = 10, img_width = 32, act_fn = nn.Tanh, act_fn_out = ll.Exp):
        super().__init__()
                
        self.img_width = img_width
        self.act_fn = act_fn(0.1)
        self.act_fn_out = act_fn_out()
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0, bias=False).requires_grad_(False)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, bias=False).requires_grad_(False)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=0, bias=False).requires_grad_(False)
        self.linear1 = nn.Linear(32, 24, bias=False)#.requires_grad_(False)
        self.linear2 = nn.Linear(24, n_cat, bias=False)#.requires_grad_(False)
        
    def init(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act_fn(x)
        x = self.maxpool(x)
        x = self.conv2(x) 
        x = self.act_fn(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.act_fn(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        #x = self.act_fn_out(x)
        return x
        
    def forward_return_all(self, x):
        y1 = self.conv1(x)
        y1 = self.act_fn(y1)
        y1 = self.maxpool(y1)
        y2 = self.conv2(y1) 
        y2 = self.act_fn(y2)
        y2 = self.maxpool(y2)
        y3 = self.conv3(y2)
        y3 = self.act_fn(y3)
        y3 = self.flatten(y3)
        y4 = self.linear1(y3)
        y4 = self.act_fn(y4)
        y5 = self.linear2(y4)
        y5 = self.act_fn(y5)
        #y5 = self.act_fn_out(y5)
        return x, y1, y2, y3, y4, y5
        
        
class Lenet_backward(pl.LightningModule):
    def __init__(self, n_cat = 10, img_width = 32, act_fn_inv = nn.LeakyReLU, act_fn_out_inv = ll.Log):
        super().__init__()

        #img dimensions        
        self.img_width0 = img_width
        self.img_width1 = self.img_width0 - (5 - 1)
        self.img_width2 = int(self.img_width1/2 - (5 - 1))
        self.img_width3 = int(self.img_width2/2 - (5 - 1))


        #layers
        self.act_fn_inv = act_fn_inv(1/0.1)
        self.act_fn_out_inv = act_fn_out_inv()
        self.maxunpool = nn.MaxUnpool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.linear2 = nn.Linear(n_cat, 24, bias=False)
        self.linear1 = nn.Linear(24, 32, bias=False)
        self.unflatten = nn.Unflatten(1, (32, 1, 1))
        self.conv3 = ll.Deconv2dfft(16, 32, kernel_size=5, image_size = self.img_width3)
        self.conv2 = ll.Deconv2dfft(6, 16, kernel_size=5, image_size = self.img_width2)
        self.conv1 = ll.Deconv2dfft(1, 6, kernel_size=5, image_size = self.img_width1)
        
    #update the convolution weight with the current state of lenet
    def update(self, lenet):
        k3_inv_fft = ll.inverse_kernel_fft(lenet.conv3.weight, self.img_width3)
        k2_inv_fft = ll.inverse_kernel_fft(lenet.conv2.weight, self.img_width2)
        k1_inv_fft = ll.inverse_kernel_fft(lenet.conv1.weight, self.img_width1)
        w_L2 = nn.Parameter(torch.linalg.pinv(lenet.linear2.weight))
        w_L1 = nn.Parameter(torch.linalg.pinv(lenet.linear1.weight))
        self.conv3.weight = k3_inv_fft
        self.conv2.weight = k2_inv_fft
        self.conv1.weight = k1_inv_fft
        self.linear2.weight = w_L2
        self.linear1.weight = w_L1
        
    def forward(self, x):
        #x=F.one_hot(x, num_classes=10)*10+1
        #x = self.act_fn_out_inv(x)
        x = F.one_hot(x, num_classes=10)*1.
        x = self.act_fn_inv(x)
        x = self.linear2(x)
        x = self.act_fn_inv(x)
        x = self.linear1(x)
        x = self.unflatten(x)
        x = self.act_fn_inv(x)
        x = self.conv3(x).real
        x = self.upsample(x)
        x = self.act_fn_inv(x)
        x = self.conv2(x).real
        x = self.upsample(x)
        x = self.act_fn_inv(x)
        x = self.conv1(x).real
        return x
        
    def forward_return_all(self, x):
        #x=F.one_hot(x, num_classes=10)*10+1
        #x = self.act_fn_out_inv(x)
        x = F.one_hot(x, num_classes=10)*5.+0.1
        x = self.act_fn_inv(x)
        y1 = self.linear2(x)
        y1 = self.act_fn_inv(y1)
        y2 = self.linear1(y1)
        y2 = self.unflatten(y2)
        y2 = self.act_fn_inv(y2)
        y3 = self.conv3(y2).real
        y3 = self.upsample(y3)
        y3 = self.act_fn_inv(y3)
        y4 = self.conv2(y3).real
        y4 = self.upsample(y4)
        y4 = self.act_fn_inv(y4)
        y5 = self.conv1(y4).real
        return x, y1, y2, y3, y4, y5


# define the Lenet LightningModule
class CNN(pl.LightningModule):
    def __init__(self, n_cat = 10, img_width = 32, act_fn = nn.Tanh, act_fn_out = nn.Softmax):
        super().__init__()
        self.save_hyperparameters()
        self.lenet = Lenet(self.hparams.n_cat, self.hparams.img_width, self.hparams.act_fn, self.hparams.act_fn_out)
        self.lenet1 = Lenet(self.hparams.n_cat, self.hparams.img_width, self.hparams.act_fn, self.hparams.act_fn_out)
        self.lenet_backward = Lenet_backward(self.hparams.n_cat, self.hparams.img_width)
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass",num_classes=n_cat)
        
    def _loss(self, batch):
        x, y = batch
        z = self.lenet(x)
        loss = F.mse_loss(z, F.one_hot(y, num_classes=10)*1.)
        #loss = F.cross_entropy(z, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.lenet(x)
        loss = self._loss(batch)
        self.accuracy(preds, y)
        self.log("train_loss", loss)
        self.log('train_acc', self.accuracy)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.lenet(x)
        loss = self._loss(batch)
        self.accuracy(preds, y)
        self.log("val_loss", loss)
        self.log('val_acc', self.accuracy)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.lenet(x)
        loss = self._loss(batch)
        self.accuracy(preds, y)
        self.log("test_loss", loss)
        self.log('test_acc', self.accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
        return [optimizer], [scheduler]
        
    def forward(self, x):
        z = self.lenet(x)
        return z
        
    def inverse(self, x):
        z = self.lenet_backward(x)
        return z
        
    def update(self):
        self.lenet_backward.update(self.lenet)

    def init(self):
        self.lenet.init()
        self.update()
        
        
    def training_step_ising(self, batch, batch_idx, N_steps):
        #parameters
        n_layer, k, stride, L, nbit, n_rep = 5, 5, 1, 10, 4, 10
        quantum=True
        beta_in, beta_final = 0.4, 50000
        db=(beta_final - beta_in)/(N_steps*n_rep)
        #propagate training batch backward
        x, y = batch
        preds_back = self.lenet_backward.forward_return_all(y)
        L_list=[1,1,1,1,1]
        #cycle over layers
        for i_layer in [3,4]:
            L=L_list[i_layer]
            #propagate training batch forward
            preds = self.lenet.forward_return_all(x)
            l_list=[self.lenet.conv1, self.lenet.conv2, self.lenet.conv3,
                self.lenet.linear1, self.lenet.linear2]
            #extract input and output of each layer
            s=preds[i_layer]
            f1z=preds_back[n_layer-i_layer-1]
            c=l_list[i_layer].weight
            #transform layer to ising configuration
            J_s, h_s, E_s, c_s = ll.layer_to_ising(s, f1z, c, k, stride, L, nbit)
            #c_s = 2*torch.randint(0,2,c_s.shape)-1
            c_s.share_memory_()
            #minimize hamiltonians
            if quantum:
                #quantum
                c_s, E_tot = ll.QuantumAnnealing(J_s, h_s, c_s)
                print("Layer",i_layer, "E_tot",E_tot)
            else:
                #classical
                for i in tqdm.tqdm(range(n_rep)):
                    beta=beta_in + (n_rep*batch_idx+i)*db
                    #multiprocess with pool
                    n_proc=min(mp.cpu_count(),c_s.shape[0])
                    args=zip(repeat(J_s), repeat(h_s), repeat(c_s), 
                             repeat(beta), range(c_s.shape[0]))
                    with mp.Pool(n_proc, initializer=torch.seed) as pool:
                        pool.starmap(ll.Metropolis_step, args)
            #update layer weights from spin configuration
            w=ll.spin_to_weight(c_s, c.shape, L, nbit)
            if i_layer==0:
                self.lenet.conv1.weight=w
            elif i_layer==1:
                self.lenet.conv2.weight=w
            elif i_layer==2:
                self.lenet.conv3.weight=w
            elif i_layer==3:
                self.lenet.linear1.weight=w
            elif i_layer==4:
                self.lenet.linear2.weight=w
        #update weights in backward network
        self.update()
        


# init the forward Lenet
img_width=32
seed=43
torch.manual_seed(seed)
cnn = CNN(n_cat = 10, img_width = img_width, act_fn = nn.LeakyReLU, act_fn_out = ll.Exp)
cnn.init()


#################################
#Dataset
DATASET=MNIST
#DATASET=FashionMNIST
# get train dataset
dataset = DATASET(root="./data", train=True, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

#split to train and validation datasets
N_train = int(len(dataset)*(1-0.1))
N_val = len(dataset) - N_train
train_set, val_set = data.random_split(dataset, [N_train, N_val])
#resize training set
indices = torch.arange(2000)
train_set = torch.utils.data.Subset(train_set, indices)

train_loader = data.DataLoader(train_set, batch_size=256, num_workers=4)
val_loader = data.DataLoader(val_set, batch_size=256, num_workers=4)

#get test dataset
dataset_test = DATASET(root="./data", train=False, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
test_loader = data.DataLoader(dataset_test, batch_size=len(dataset_test), num_workers=4)


#################################
#Train

log_folder="./saved_models/lenet/"
os.makedirs(log_folder, exist_ok=True)

train = len(sys.argv)>1 and sys.argv[1] == 'train'
load = len(sys.argv)>1 and sys.argv[1] == 'load'
load_sg = len(sys.argv)>1 and sys.argv[1] == 'load_sg'
load_sq = len(sys.argv)>1 and sys.argv[1] == 'load_sq'

cnn.train()
cnn.cuda()
trainer = pl.Trainer(default_root_dir=log_folder, max_epochs=100, accelerator="gpu", callbacks=[EarlyStopping(monitor="val_acc", mode="max", patience=5), LearningRateMonitor("epoch")])
# train the model
if train:
    trainer.fit(model=cnn, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Test model on validation and test set
    val_result = trainer.test(cnn, val_loader, verbose=False)
    test_result = trainer.test(cnn, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    print("Test results:")
    print(result["test"])
    print(result["val"])
    print(cnn.lenet.linear1.weight.min().item(),cnn.lenet.linear1.weight.max().item())
    exit()

#visualize log in tensorboard: tensorboard --logdir ./saved_models/lenet/



#################################   
#load saved weights and test

# load checkpoint
if load:
    checkpoint = log_folder+"lightning_logs/version_392/checkpoints/epoch=19-step=160.ckpt"
    cnn = CNN.load_from_checkpoint(checkpoint)

    cnn.lenet1=copy.deepcopy(cnn.lenet)
    #nn.init.xavier_uniform_(cnn.lenet.conv3.weight)
    #nn.init.xavier_uniform_(cnn.lenet.linear1.weight)
    #nn.init.xavier_uniform_(cnn.lenet.linear2.weight)

if load_sg:
    cnn = torch.load(log_folder+'SG/model.pt')

#evaluate model on cpu or gpu
cnn.cpu()
#cnn.cuda()
#set net to evaluation mode
cnn.eval()
#update weights of backward net
cnn.update()

#load weights trained in the squared network
if load_sq:
    w1=torch.load(path+'SG/w1.pt')
    w2=torch.load(path+'SG/w2.pt')
    c1=torch.load(path+'SG/c1.pt')
    c2=torch.load(path+'SG/c2.pt')
    c3=torch.load(path+'SG/c3.pt')
    cnn.lenet.linear1.weight=nn.Parameter(w1)
    cnn.lenet.linear2.weight=nn.Parameter(w2)
    cnn.lenet.conv1.weight=nn.Parameter(c1)
    cnn.lenet.conv2.weight=nn.Parameter(c2)
    cnn.lenet.conv3.weight=nn.Parameter(c3)


# Test model on validation and test set
test_result = trainer.test(cnn, test_loader, verbose=False)
print("Test results before:")
print(test_result)


################################
#Train as spin glass with Metropolis algorithm
train_loader = data.DataLoader(train_set, batch_size=len(train_set), num_workers=4)
with torch.no_grad():
    cnn.cpu()
    train_iterator=iter(train_loader)
    #train_features, train_labels = next(train_iterator)
    Nsteps=len(train_iterator)
    Ncycles=10
    max_acc=test_result[0]['test_acc']
    i_max_acc=0
    k_max_acc=0
    for k in range(Ncycles):
        for i in range(Nsteps):
            train_features, train_labels = next(train_iterator)
            print("step "+str(i+1)+"/"+str(Nsteps)+" cycle "+str(k+1)+"/"+str(Ncycles))
            cnn.training_step_ising([train_features.to(cnn.device), train_labels.to(cnn.device)], k*Nsteps+i, Nsteps*Ncycles)
            # Test model on validation and test set
            test_result = trainer.test(cnn, test_loader, verbose=False)
            #print("Test results after:")
            print(test_result)
            if test_result[0]['test_acc']>max_acc:
                max_acc=test_result[0]['test_acc']
                i_max_acc=i
                k_max_acc=k
                torch.save(cnn, log_folder+'SG/model.pt')
        train_iterator=iter(train_loader)
print("max_acc", max_acc, k_max_acc+1, i_max_acc+1)

#load max accuracy weights
cnn = torch.load(log_folder+'SG/model.pt')

# Test model on validation and test set
test_result = trainer.test(cnn, test_loader, verbose=False)
print("test SG")
print(test_result)

cnn.lenet=copy.deepcopy(cnn.lenet1)
# Test model on validation and test set
test_result = trainer.test(cnn, test_loader, verbose=False)
print("test NN")
print(test_result)

    
#test_iterator=iter(test_loader)
#test_features, test_labels = next(test_iterator)   
#out=cnn(test_features)
#res= [(torch.argmax(pred).item(), y.item()) for pred, y in zip(out, test_labels)]
#print(res)
#for o in out[:10]:
#    print(o.data)




