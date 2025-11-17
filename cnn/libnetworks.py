import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging
import libannealing as la
import copy
import signal


#disable tracebackon ctrl+c
signal.signal(signal.SIGINT, lambda x, y: sys.exit())

#disable logging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

#do not use gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#disable scientific notation
torch.set_printoptions(sci_mode=False, threshold=20000)


#############################
#network

class lenet_small(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.maxpool = nn.MaxPool2d(2)
        self.spectralpool = la.SpectralPool2d(1/2)
        self.flatten = nn.Flatten()
        self.L0 = nn.Conv2d(in_channel, 6, kernel_size=5, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Conv2d(6, 12, kernel_size=5, bias=False).requires_grad_(1 in l_train)
        self.L2 = nn.Conv2d(12, 24, kernel_size=5, bias=False).requires_grad_(2 in l_train)
        self.L3 = nn.Linear(24, 18, bias=False).requires_grad_(3 in l_train)
        self.L4 = nn.Linear(18, n_cat, bias=False).requires_grad_(4 in l_train)
        self.layer_list=[self.L0, self.L1, self.L2, self.L3, self.L4]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.xavier_uniform_(self.L3.weight)
        nn.init.xavier_uniform_(self.L4.weight)
        
    def forward_return_all(self, x):
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        y1 = self.spectralpool(y1)
        y2na = self.L1(y1) 
        y2 = self.act_fn(y2na)
        y2 = self.spectralpool(y2)
        y3na = self.L2(y2)
        y3 = self.act_fn(y3na)
        y3 = self.flatten(y3)
        y4na = self.L3(y3)
        y4 = self.act_fn(y4na)
        y5na = self.L4(y4)
        y5 = self.act_fn(y5na)
        return x, y1, y2, y3, y4, y5, y1na, y2na, y3na, y4na, y5na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-6] 
        
class lenet_small_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        self.alpha=alpha
        k_size = [5,5,5]
        pool_size = [1,2,2]
        #img dimensions        
        self.img_widths=[img_width]
        for i in range(len(k_size)):
            width = int(self.img_widths[i]/pool_size[i]-(k_size[i]-1))
            self.img_widths.append(width)
        #layers
        self.unpool = nn.MaxUnpool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.spectralunpool = la.SpectralPool2d(2)
        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.L4 = nn.Linear(n_cat, 18, bias=False)
        self.L3 = nn.Linear(18, 24, bias=False)
        self.unflatten = nn.Unflatten(1, (24, 1, 1))
        self.L2 = la.Deconv2dfft(12, 24, kernel_size=k_size[2], image_size = self.img_widths[3])
        self.L1 = la.Deconv2dfft(6, 12, kernel_size=k_size[1], image_size = self.img_widths[2])
        self.L0 = la.Deconv2dfft(1, 6, kernel_size=k_size[0], image_size = self.img_widths[1])
        self.layer_list=[self.L0, self.L1, self.L2]
        
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L2.inverse(network.L2.weight)
        self.L1.inverse(network.L1.weight)
        self.L0.inverse(network.L0.weight)
        self.L4.weight = nn.Parameter(torch.linalg.pinv(network.L4.weight))
        self.L3.weight = nn.Parameter(torch.linalg.pinv(network.L3.weight))
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        y1 = self.L4(x)
        y2 = self.L3(y1)
        y2 = self.unflatten(y2)
        y3 = self.L2(y2)
        y3 = self.spectralunpool(y3)*10
        y4 = self.L1(y3)
        y4 = self.spectralunpool(y4)*10
        y5 = self.L0(y4)
        return x, y1, y2, y3, y4, y5


#########################

class lenet(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.maxpool = nn.MaxPool2d(2, return_indices=True)
        self.spectralpool = la.SpectralPool2d(1/2)
        self.flatten = nn.Flatten()
        self.L0 = nn.Conv2d(in_channel, 6, kernel_size=5, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Conv2d(6, 16, kernel_size=5, bias=False).requires_grad_(1 in l_train)
        self.L2 = nn.Conv2d(16, 120, kernel_size=5, bias=False).requires_grad_(2 in l_train)
        self.L3 = nn.Linear(120, 84, bias=False).requires_grad_(3 in l_train)
        self.L4 = nn.Linear(84, n_cat, bias=False).requires_grad_(4 in l_train)
        self.layer_list=[self.L0, self.L1, self.L2, self.L3, self.L4]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.xavier_uniform_(self.L3.weight)
        nn.init.xavier_uniform_(self.L4.weight)
        
    def forward_return_all(self, x):
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        y1 = self.spectralpool(y1)
        y2na = self.L1(y1) 
        y2 = self.act_fn(y2na)
        y2 = self.spectralpool(y2)
        y3na = self.L2(y2)
        y3 = self.act_fn(y3na)
        y3 = self.flatten(y3)
        y4na = self.L3(y3)
        y4 = self.act_fn(y4na)
        y5na = self.L4(y4)
        y5 = self.act_fn(y5na)
        return x, y1, y2, y3, y4, y5, y1na, y2na, y3na, y4na, y5na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-6]    
        
class lenet_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        k_size = [5,5,5]
        pool_size = [1,2,2]
        #img dimensions        
        self.img_widths=[img_width]
        for i in range(len(k_size)):
            width = int(self.img_widths[i]/pool_size[i]-(k_size[i]-1))
            self.img_widths.append(width)
        #layers
        self.spectralpool = la.SpectralPool2d(2)
        self.L4 = nn.Linear(n_cat, 84, bias=False)
        self.L3 = nn.Linear(84, 120, bias=False)
        self.unflatten = nn.Unflatten(1, (120, 1, 1))
        self.L2 = la.Deconv2dfft(16, 120, kernel_size=k_size[2], image_size = self.img_widths[3])
        self.L1 = la.Deconv2dfft(6, 16, kernel_size=k_size[1], image_size = self.img_widths[2])
        self.L0 = la.Deconv2dfft(1, 6, kernel_size=k_size[0], image_size = self.img_widths[1])
        
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L4.weight = nn.Parameter(torch.linalg.pinv(network.L4.weight))
        self.L3.weight = nn.Parameter(torch.linalg.pinv(network.L3.weight))
        self.L2.inverse(network.L2.weight)
        self.L1.inverse(network.L1.weight)
        self.L0.inverse(network.L0.weight)
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        y1 = self.L4(x)
        y2 = self.L3(y1)
        y2 = self.unflatten(y2)
        y3 = self.L2(y2)
        y3 = self.spectralpool(y3)*100
        y4 = self.L1(y3)
        y4 = self.spectralpool(y4)*100
        y5 = self.L0(y4)
        return x, y1, y2, y3, y4, y5

#########################

class linear(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.maxpool = nn.MaxPool2d(4)
        self.spectralpool = la.SpectralPool2d(1/4)
        self.flatten = nn.Flatten()
        self.L0 = nn.Linear(64*in_channel, 80, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Linear(80, 96, bias=False).requires_grad_(1 in l_train)
        self.L2 = nn.Linear(96, 112, bias=False).requires_grad_(2 in l_train)
        self.L3 = nn.Linear(112, 24, bias=False).requires_grad_(3 in l_train)
        self.L4 = nn.Linear(24, n_cat, bias=False).requires_grad_(4 in l_train)
        self.layer_list=[self.L0, self.L1, self.L2, self.L3, self.L4]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.xavier_uniform_(self.L3.weight)
        nn.init.xavier_uniform_(self.L4.weight)
        
    def forward_return_all(self, x):
        x = self.spectralpool(x)
        x = self.flatten(x)
        y1 = self.L0(x)
        y1 = self.act_fn(y1)
        y2 = self.L1(y1)
        y2 = self.act_fn(y2)
        y3 = self.L2(y2)
        y3 = self.act_fn(y3)
        y4 = self.L3(y3)
        y4 = self.act_fn(y4)
        y5 = self.L4(y4)
        #y5 = self.act_fn(y5.clamp(-L0,L1))
        return x, y1, y2, y3, y4, y5
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-1]
        
        
class linear_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.act_fn_inv = nn.LeakyReLU(alpha)
        self.unflatten = nn.Unflatten(1, (1, 8, 8))
        self.spectralpool = la.SpectralPool2d(4)
        self.L0 = nn.Linear(80, 64, bias=False)
        self.L1 = nn.Linear(96, 80, bias=False)
        self.L2 = nn.Linear(112, 96, bias=False)
        self.L3 = nn.Linear(24, 112, bias=False)
        self.L4 = nn.Linear(n_cat, 24, bias=False)
  
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L4.weight = nn.Parameter(torch.linalg.pinv(network.L4.weight))
        self.L3.weight = nn.Parameter(torch.linalg.pinv(network.L3.weight))
        self.L2.weight = nn.Parameter(torch.linalg.pinv(network.L2.weight))
        self.L1.weight = nn.Parameter(torch.linalg.pinv(network.L1.weight))
        self.L0.weight = nn.Parameter(torch.linalg.pinv(network.L0.weight))
        
#    #used with metropolis_act
#    def forward_return_all(self, x):
#        x = F.one_hot(x, num_classes=10)*10.
#        #x = self.act_fn_inv(x)
#        y1 = self.L4(x)
#        y2 = self.act_fn_inv(y1)
#        y2 = self.L3(y2)
#        y3 = self.act_fn_inv(y2)
#        y3 = self.L2(y3)
#        y4 = self.act_fn_inv(y3)
#        y4 = self.L1(y4)
#        y5 = self.act_fn_inv(y4)
#        y5 = self.L0(y5)
#        return x, y1, y2, y3, y4, y5
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        #x = self.act_fn_inv(x)
        y1 = self.L4(x)
        #y1 = self.act_fn_inv(y1)
        y2 = self.L3(y1)
        #y2 = self.act_fn_inv(y2)
        y3 = self.L2(y2)
        #y3 = self.act_fn_inv(y3)
        y4 = self.L1(y3)
        #y4 = self.act_fn_inv(y4)
        y5 = self.L0(y4)
        y5 = self.unflatten(y5)
        y5 = self.spectralpool(y5)
        return x, y1, y2, y3, y4, y5
        
#########################

class cnn_no_pool(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.L0 = nn.Conv2d(in_channel, 6, kernel_size=5, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Conv2d(6, 16, kernel_size=5, bias=False).requires_grad_(1 in l_train)
        self.L2 = nn.Conv2d(16, 32, kernel_size=5, bias=False).requires_grad_(2 in l_train)
        self.L3 = nn.Linear(512, 24, bias=False).requires_grad_(3 in l_train)
        self.L4 = nn.Linear(24, n_cat, bias=False).requires_grad_(4 in l_train)
        self.layer_list=[self.L0, self.L1, self.L2, self.L3, self.L4]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.xavier_uniform_(self.L3.weight)
        nn.init.xavier_uniform_(self.L4.weight)
        
    def forward_return_all(self, x):
        x = self.maxpool(x)
        y1 = self.L0(x)
        y1 = self.act_fn(y1)
        y2 = self.L1(y1) 
        y2 = self.act_fn(y2)
        y3 = self.L2(y2)
        y3 = self.act_fn(y3)
        y3 = self.flatten(y3)
        y4 = self.L3(y3)
        y4 = self.act_fn(y4)
        y5 = self.L4(y4)
        y5 = self.act_fn(y5)
        return x, y1, y2, y3, y4, y5
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-1]
        
        
class cnn_no_pool_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        k_size = [5,5,5]
        pool_size = [2,1,1]
        #img dimensions        
        self.img_widths=[img_width]
        for i in range(len(k_size)):
            width = int(self.img_widths[i]/pool_size[i]-(k_size[i]-1))
            self.img_widths.append(width)
        #layers
        self.act_fn_inv = nn.LeakyReLU(1/alpha)
        self.unpool = nn.MaxUnpool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.L4 = nn.Linear(n_cat, 24, bias=False)
        self.L3 = nn.Linear(24, 512, bias=False)
        self.unflatten = nn.Unflatten(1, (32, 4, 4))
        self.L2 = la.Deconv2dfft(16, 32, kernel_size=k_size[2], image_size = self.img_widths[3])
        self.L1 = la.Deconv2dfft(6, 16, kernel_size=k_size[1], image_size = self.img_widths[2])
        self.L0 = la.Deconv2dfft(1, 6, kernel_size=k_size[0], image_size = self.img_widths[1])
        
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L2.inverse(network.L2.weight)
        self.L1.inverse(network.L1.weight)
        self.L0.inverse(network.L0.weight)
        w_L1 = nn.Parameter(torch.linalg.pinv(network.L4.weight))
        w_L0 = nn.Parameter(torch.linalg.pinv(network.L3.weight))
        self.L4.weight = w_L1
        self.L3.weight = w_L0
        
    def forward_return_all(self, x):
        L=0.5
        x = F.one_hot(x, num_classes=10)*1.
        x = self.act_fn_inv(x).clamp(-L,100*L)
        y1 = self.L4(x)
        y1 = self.act_fn_inv(y1).clamp(-L,100*L)
        y2 = self.L3(y1)
        y2 = self.unflatten(y2)
        y2 = self.act_fn_inv(y2).clamp(-L,100*L)
        y3 = self.L2(y2)
        y3 = self.act_fn_inv(y3).clamp(-L,100*L)
        y4 = self.L1(y3)
        y4 = self.act_fn_inv(y4).clamp(-L,100*L)
        y5 = self.L0(y4)
        return x, y1, y2, y3, y4, y5
        
#########################

class single_layer(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.flatten = nn.Flatten()
        self.L0 = nn.Linear(1024*in_channel, n_cat, bias=False).requires_grad_(0 in l_train)
        self.layer_list=[self.L0]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        
    def forward_return_all(self, x):
        x = self.flatten(x)
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        return x, y1, y1na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-2]
        
        
class single_layer_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.act_fn_inv = nn.LeakyReLU(alpha)
        self.L0 = nn.Linear(n_cat, 1024, bias=False)
        self.layer_list=[self.L0]
  
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L0.weight = nn.Parameter(torch.linalg.pinv(network.L0.weight))
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        #x = self.act_fn_inv(x)
        y1 = self.L0(x)
        return x, y1
        
#########################

class single_layer_small(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(8)
        self.L0 = nn.Linear(16*in_channel, n_cat, bias=False).requires_grad_(0 in l_train)
        self.layer_list=[self.L0]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        
    def forward_return_all(self, x):
        x = self.maxpool(x)
        x = self.flatten(x)
        y1 = self.L0(x)
        y1 = self.act_fn(y1)
        return x, y1
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-1]
        
        
class single_layer_small_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.act_fn_inv = nn.LeakyReLU(1/alpha)
        self.L0 = nn.Linear(n_cat, 16, bias=False)
  
    #update the convolution weight with the current state of network
    def update(self, network):
        w_L0 = nn.Parameter(torch.linalg.pinv(network.L0.weight))
        self.L0.weight = w_L0
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*1.
        x = self.act_fn_inv(x)#.clamp(-L,100*L)
        y1 = self.L0(x)
        return x, y1
        
#########################

class single_layer_very_small(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.flatten = nn.Flatten()
        #self.maxpool = nn.MaxPool2d((8,16))
        self.spectralpool = la.SpectralPool2d((1/8,1/16))
        self.L0 = nn.Linear(8*in_channel, n_cat, bias=False).requires_grad_(0 in l_train)
        self.layer_list=[self.L0]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        
    def forward_return_all(self, x):
        #x = self.maxpool(x)
        x = self.spectralpool(x)
        x = self.flatten(x)
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        return x, y1, y1na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-2]
        
        
class single_layer_very_small_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.act_fn_inv = nn.LeakyReLU(1/alpha)
        self.L0 = nn.Linear(n_cat, 8, bias=False)
        self.layer_list=[self.L0]
  
    #update the convolution weight with the current state of network
    def update(self, network):
        w_L0 = nn.Parameter(torch.linalg.pinv(network.L0.weight))
        self.L0.weight = w_L0
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*2.
        #x = self.act_fn_inv(x)
        y1 = self.L0(x)
        return x, y1
        
#########################

class two_layer(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.flatten = nn.Flatten()
        self.spectralpool = la.SpectralPool2d(1/4)
        #self.L0 = nn.Linear(256*in_channel, 64, bias=False).requires_grad_(0 in l_train)
        #self.L1 = nn.Linear(64, n_cat, bias=False).requires_grad_(0 in l_train)
        self.L0 = nn.Linear(64*in_channel, 128, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Linear(128, n_cat, bias=False).requires_grad_(1 in l_train)
        self.layer_list=[self.L0, self.L1]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        
    def forward_return_all(self, x):
        x = self.spectralpool(x)
        x = self.flatten(x)
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        y2na = self.L1(y1)
        y2 = self.act_fn(y2na)
        return x, y1, y2, y1na, y2na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-1]
        
class two_layer_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.spectralpool = la.SpectralPool2d(4)
        self.act_fn_inv = nn.LeakyReLU(1/alpha)
        #self.L0 = nn.Linear(64, 256, bias=False)
        #self.L1 = nn.Linear(n_cat, 64, bias=False)
        self.L0 = nn.Linear(128, 64, bias=False)
        self.L1 = nn.Linear(n_cat, 128, bias=False)
        self.layer_list=[self.L0, self.L1]
  
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L0.weight = nn.Parameter(torch.linalg.pinv(network.L0.weight))
        self.L1.weight = nn.Parameter(torch.linalg.pinv(network.L1.weight))
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        #x = self.act_fn_inv(x)
        y1 = self.L1(x)
        #y1 = self.act_fn_inv(y1)
        y2 = self.L0(y1)
        return x, y1, y2
        
class two_layer_bw1(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.spectralpool = la.SpectralPool2d(4)
        self.act_fn_inv = nn.LeakyReLU(1/alpha)
        self.L0 = nn.Linear(256, 64, bias=False)
        self.L1 = nn.Linear(n_cat, 256, bias=False)
        self.layer_list=[self.L0, self.L1]
  
    #update the convolution weight with the current state of network
    def update(self, network):
        for i in range(len(network.layer_list)):
            w=network.layer_list[i].weight
            if w.shape[1]>w.shape[0]:
                n=w.shape[1]-w.shape[0]
                diag=torch.diag(torch.ones(n),w.shape[0])[:-w.shape[0]]
                w_square=torch.cat((w,diag))
                w_inv=torch.linalg.pinv(w_square)
            else:
                w_inv = torch.linalg.pinv(w)
            self.layer_list[i].weight = nn.Parameter(w_inv)
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        y1 = torch.cat((x, torch.zeros(x.shape[0],self.L1.weight.shape[1]-x.shape[1])), 1)
        #x = self.act_fn_inv(x)
        y1 = self.L1(y1)
        y2 = torch.cat((y1, torch.zeros(y1.shape[0],self.L0.weight.shape[1]-y1.shape[1])), 1)
        #y1 = self.act_fn_inv(y1)
        y2 = self.L0(y2)
        return x, y1, y2
        
#########################

class three_layer(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.flatten = nn.Flatten()
        self.spectralpool = la.SpectralPool2d(1/4)
        self.L0 = nn.Linear(64*in_channel, 128, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Linear(128, 64, bias=False).requires_grad_(1 in l_train)
        self.L2 = nn.Linear(64, n_cat, bias=False).requires_grad_(2 in l_train)
        self.layer_list=[self.L0, self.L1, self.L2]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        
    def forward_return_all(self, x):
        x = self.spectralpool(x)
        x = self.flatten(x)
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        y2na = self.L1(y1)
        y2 = self.act_fn(y2na)
        y3na = self.L2(y2)
        y3 = self.act_fn(y3na)
        return x, y1, y2, y3, y1na, y2na, y3na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-4]
        
class three_layer_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.spectralpool = la.SpectralPool2d(4)
        self.unflatten = nn.Unflatten(1, (1, 8, 8))
        #self.act_fn_inv = nn.LeakyReLU(1/alpha)
        self.L0 = nn.Linear(128, 64, bias=False)
        self.L1 = nn.Linear(64, 128, bias=False)
        self.L2 = nn.Linear(n_cat, 64, bias=False)
        self.layer_list=[self.L0, self.L1, self.L2]
  
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L0.weight = nn.Parameter(torch.linalg.pinv(network.L0.weight))
        self.L1.weight = nn.Parameter(torch.linalg.pinv(network.L1.weight))
        self.L2.weight = nn.Parameter(torch.linalg.pinv(network.L2.weight))
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        y1 = self.L2(x)
        y2 = self.L1(y1)*2
        y3 = self.L0(y2)*10
        return x, y1, y2, y3
        
#########################

class two_layer_very_small(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((8,16))
        self.L0 = nn.Linear(8*in_channel, 8, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Linear(8, n_cat, bias=False).requires_grad_(1 in l_train)
        self.layer_list=[self.L0, self.L1]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        
    def forward_return_all(self, x):
        x = self.maxpool(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        y1 = self.L0(x)
        print(y1.shape)
        y1 = self.act_fn(y1)
        y2 = self.L1(y1)
        print(y2.shape)
        y2 = self.act_fn(y2)
        exit()
        return x, y1, y2
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-1]
        
        
class two_layer_very_small_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.act_fn_inv = nn.LeakyReLU(1/alpha)
        self.L0 = nn.Linear(8, 8, bias=False)
        self.L1 = nn.Linear(n_cat, 8, bias=False)
  
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L0.weight = nn.Parameter(torch.linalg.pinv(network.L0.weight))
        self.L1.weight = nn.Parameter(torch.linalg.pinv(network.L1.weight))
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*1.
        x = self.act_fn_inv(x)#.clamp(-L,100*L)
        y1 = self.L1(x)
        y1 = self.act_fn_inv(y1)
        y2 = self.L0(y1)
        return x, y1, y2
        
#########################

class single_layer_conv(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.spectralpool = la.SpectralPool2d(1/4)
        self.spectralpool1 = la.SpectralPool2d(1/2)
        self.flatten = nn.Flatten()
        self.L0 = nn.Conv2d(in_channel, 10, kernel_size=8, bias=False).requires_grad_(0 in l_train)
        #self.L4 = nn.Linear(18, n_cat, bias=False).requires_grad_(4 in l_train)
        self.layer_list=[self.L0]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        
    def forward_return_all(self, x):
        #x = F.pad(x, (2,2,2,2))
        #x = self.maxpool(x)
        x = self.spectralpool(x)
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        #y1 = self.maxpool(y1)
        y1 = self.flatten(y1)
        return x, y1, y1na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-2] 
        
class single_layer_conv_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.spectralpool = la.SpectralPool2d(4)
        self.L0 = la.Deconv2dfft(1, 10, kernel_size=8, image_size = 1)
        self.unflatten = nn.Unflatten(1, (10, 1, 1))
        
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L0.inverse(network.L0.weight)
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        x = self.unflatten(x)
        #x = self.upsample(x)
        #x = self.spectralpool(x)
        y1 = self.L0(x)
        return x, y1

#########################

class single_layer_conv_small(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.maxpool = nn.MaxPool2d(4)
        self.spectralpool = la.SpectralPool2d(1/8)
        self.act_fn = nn.LeakyReLU(alpha)
        self.flatten = nn.Flatten()
        self.L0 = nn.Conv2d(in_channel, 10, kernel_size=4, bias=False).requires_grad_(0 in l_train)
        #self.L4 = nn.Linear(18, n_cat, bias=False).requires_grad_(4 in l_train)
        self.layer_list=[self.L0]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        
    def forward_return_all(self, x):
        x = self.spectralpool(x)
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        y1 = self.flatten(y1)
        return x, y1, y1na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-2] 
        
class single_layer_conv_small_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.spectralpool = la.SpectralPool2d(8)
        self.L0 = la.Deconv2dfft(1, 10, kernel_size=4, image_size = 1)
        self.unflatten = nn.Unflatten(1, (10, 1, 1))
        
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L0.inverse(network.L0.weight)
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        x = self.unflatten(x)
        #x = self.upsample(x)
        #x = self.spectralpool(x)
        y1 = self.L0(x)
        return x, y1
        
#########################

class two_layer_conv(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.spectralpool = la.SpectralPool2d(1/4)
        self.spectralpool1 = la.SpectralPool2d(1/2)
        self.flatten = nn.Flatten()
        self.L0 = nn.Conv2d(in_channel, 20, kernel_size=5, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Conv2d(20, 10, kernel_size=4, bias=False).requires_grad_(1 in l_train)
        #self.L4 = nn.Linear(18, n_cat, bias=False).requires_grad_(4 in l_train)
        self.layer_list=[self.L0, self.L1]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        
    def forward_return_all(self, x):
        #x = F.pad(x, (2,2,2,2))
        #x = self.maxpool(x)
        x = self.spectralpool(x)
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        y2na = self.L1(y1)
        y2 = self.act_fn(y2na)
        #y1 = self.maxpool(y1)
        y2 = self.flatten(y2)
        return x, y1, y2, y1na, y2na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-3] 
        
class two_layer_conv_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.spectralpool = la.SpectralPool2d(4)
        self.L0 = la.Deconv2dfft(1, 20, kernel_size=5, image_size = 4)
        self.L1 = la.Deconv2dfft(20, 10, kernel_size=4, image_size = 1)
        self.unflatten = nn.Unflatten(1, (10, 1, 1))
        
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L0.inverse(network.L0.weight)
        self.L1.inverse(network.L1.weight)
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        x = self.unflatten(x)
        #x = self.upsample(x)
        #x = self.spectralpool(x)
        y1 = self.L1(x)*10
        y2 = self.L0(y1)
        y2 = self.spectralpool(y2)
        return x, y1, y2

#########################

class two_layer_conv_small(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.spectralpool = la.SpectralPool2d(1/8)
        self.flatten = nn.Flatten()
        self.L0 = nn.Conv2d(in_channel, 4, kernel_size=3, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Conv2d(4, 10, kernel_size=2, bias=False).requires_grad_(1 in l_train)
        #self.L4 = nn.Linear(18, n_cat, bias=False).requires_grad_(4 in l_train)
        self.layer_list=[self.L0, self.L1]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        
    def forward_return_all(self, x):
        x = self.spectralpool(x)
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        y2na = self.L1(y1)
        y2 = self.act_fn(y2na)
        y2 = self.flatten(y2)
        return x, y1, y2, y1na, y2na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-3] 
        
class two_layer_conv_small_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        #layers
        self.spectralpool = la.SpectralPool2d(8)
        self.L0 = la.Deconv2dfft(1, 4, kernel_size=3, image_size = 2)
        self.L1 = la.Deconv2dfft(4, 10, kernel_size=2, image_size = 1)
        self.unflatten = nn.Unflatten(1, (10, 1, 1))
        
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L0.inverse(network.L0.weight)
        self.L1.inverse(network.L1.weight)
        
    def forward_return_all(self, x):
        x = F.one_hot(x, num_classes=10)*10.
        x = self.unflatten(x)
        #x = self.upsample(x)
        #x = self.spectralpool(x)
        y1 = self.L1(x)*10
        y2 = self.L0(y1)
        y2 = self.spectralpool(y2)
        return x, y1, y2

#########################
        
class three_layer_conv(pl.LightningModule):
    def __init__(self, n_cat, in_channel, alpha, l_train):
        super().__init__()
        self.act_fn = nn.LeakyReLU(alpha)
        self.spectralpool = la.SpectralPool2d(1/2)
        self.flatten = nn.Flatten()
        self.L0 = nn.Conv2d(in_channel, 10, kernel_size=5, bias=False).requires_grad_(0 in l_train)
        self.L1 = nn.Conv2d(10, 30, kernel_size=5, bias=False).requires_grad_(1 in l_train)
        self.L2 = nn.Conv2d(30, 10, kernel_size=5, bias=False).requires_grad_(2 in l_train)
        self.layer_list=[self.L0, self.L1, self.L2]
        
    def init(self):
        nn.init.xavier_uniform_(self.L0.weight)
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        
    def forward_return_all(self, x):
        y1na = self.L0(x)
        y1 = self.act_fn(y1na)
        y1 = self.spectralpool(y1)
        y2na = self.L1(y1) 
        y2 = self.act_fn(y2na)
        y2 = self.spectralpool(y2)
        y3na = self.L2(y2)
        y3 = self.act_fn(y3na)
        y3 = self.flatten(y3)
        return x, y1, y2, y3, y1na, y2na, y3na
        
    def forward(self, x):
        y = self.forward_return_all(x)
        return y[-4]    
        
class three_layer_conv_bw(pl.LightningModule):
    def __init__(self, n_cat, img_width, alpha):
        super().__init__()
        self.alpha = alpha
        k_size = [5,5,5]
        pool_size = [1,2,2]
        #img dimensions        
        self.img_widths=[img_width]
        for i in range(len(k_size)):
            width = int(self.img_widths[i]/pool_size[i]-(k_size[i]-1))
            self.img_widths.append(width)
        #layers
        #self.act_fn_inv = nn.LeakyReLU(1/alpha)
        self.spectralpool = la.SpectralPool2d(2)
        self.unflatten = nn.Unflatten(1, (10, 1, 1))
        self.L2 = la.Deconv2dfft(30, 10, kernel_size=k_size[2], image_size = self.img_widths[3])
        self.L1 = la.Deconv2dfft(10, 30, kernel_size=k_size[1], image_size = self.img_widths[2])
        self.L0 = la.Deconv2dfft(1, 10, kernel_size=k_size[0], image_size = self.img_widths[1])
        
    #update the convolution weight with the current state of network
    def update(self, network):
        self.L2.inverse(network.L2.weight)
        self.L1.inverse(network.L1.weight)
        self.L0.inverse(network.L0.weight)
        
    def forward_return_all(self, x):
        L=0.5
        x = F.one_hot(x, num_classes=10)*10.
        x = self.unflatten(x)
        y1 = self.L2(x)
        y1 = self.spectralpool(y1)*10
        #y1 = self.act_fn_inv(y1)*10
        y2 = self.L1(y1)
        y2 = self.spectralpool(y2)*10
        #y2 = self.act_fn_inv(y2)*10
        y3 = self.L0(y2)
        return x, y1, y2, y3
