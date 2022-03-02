from casadi.casadi import forward
from numpy.lib.type_check import real
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from scipy.linalg import pinv2
from casadi import *
import casadi

class EstimatorDummy:
    def __init__(self):
        pass

    def forward(self, x, u, t, train=False):
        return None
    
    def train(self, t, data):
        pass


class EstimatorNN:
    def __init__(self, input_size, hidden_size, output_size, lr = 1e-4):
        self.e_f = NN(input_size, hidden_size, output_size)
        self.e_g = NN(input_size, hidden_size, output_size)
        self.lr = lr
        self.opts = {'e_f': torch.optim.Adam(self.e_f.model.parameters(), lr = self.lr), 
                     'e_g': torch.optim.Adam(self.e_g.model.parameters(), lr = self.lr)}

    def forward(self, x, u, t, train=False):
        ef = self.e_f.forward(x, train)
        eg = self.e_g.forward(x, train)
        return ef + eg * u
    
    def train(self, t, data):
        # Train estimator
        sample = data.get_D()

        for x_i, k_i, dhe_real_i in zip(sample.x, sample.k, sample.dhe_real):
            S_i = self.forward(x_i, k_i, None, train=True)

            loss = F.mse_loss(torch.tensor(dhe_real_i),S_i)

            self.e_f.model.zero_grad()
            self.e_g.model.zero_grad()

            loss.backward()
            self.opts['e_f'].step()
            self.opts['e_g'].step()
        

class NN:
    def __init__(self, input_size, hidden_size, output_size = 1):
        self.model = nn.Sequential(
                                  nn.Linear(input_size, hidden_size),                   nn.Sigmoid(),
                                  nn.Linear(hidden_size, hidden_size),                  nn.Sigmoid(),
#                                  nn.Linear(hidden_size, hidden_size),                  nn.Sigmoid(),
                                  nn.Linear(hidden_size, output_size)
                                )
        
        self.model.apply(self.weights_init)
        

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.5)
            #m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)


    def forward(self, x, train = False):
        x = torch.from_numpy(np.array(x)).float()
        #print(x)
        z = self.model(x)
        
        if not train:
            z = z.detach().float().item()
            
        return z

#########################################################################################################

#########################################################################################################

class ELM: 
    def __init__(self, input_size, hidden_size = 100, output_size = 1):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        self.H = nn.Sequential(nn.Linear(input_size, self.hidden_size),                   nn.Sigmoid(),
                               )
        
        self.H.requires_grad_(False)
        
        self.elm = nn.Linear(self.hidden_size, self.output_size)
        
        self.H.apply(self.weights_init)
        self.elm.apply(self.weights_init)
        
        self.model = nn.Sequential(self.H, self.elm)
        

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # m.weight.data.fill_(0.0)
            m.weight.data.normal_(0.0, 0.01)
            m.bias.data.fill_(0.0)


    def forward(self, x, train = False):
        x = torch.from_numpy(np.array(x)).float()
        
        z = self.model(x)
        
        if not train:
            z = z.detach().float().item()
            
        return z
    
    
class EstimatorELM:
    def __init__(self, input_size, hidden_size, output_size, time_th, dt, lrate_pre = 1e-4, lrate_post = 5e-3 ):
        
        self.time_th = time_th
        self.dt = dt
        
        self.lr_pre = lrate_pre
        self.lr_post = lrate_post
        
        self.first_trained = False
        
        self.hidden_size = hidden_size
        self.e_f = ELM(input_size, hidden_size, output_size)
        self.e_g = ELM(input_size, hidden_size, output_size)
        
        self.opts_pre = {'e_f': torch.optim.Adam(self.e_f.model.parameters(), lr = self.lr_pre), 
                         'e_g': torch.optim.Adam(self.e_g.model.parameters(), lr = self.lr_pre)}
        
        self.opts_post = {'e_f': torch.optim.Adam(self.e_f.model.parameters(), lr = self.lr_post), 
                          'e_g': torch.optim.Adam(self.e_g.model.parameters(), lr = self.lr_post)}
        

    def forward(self, x, u, t, train=False):
        if self.first_trained: # t / self.dt > self.time_th and  #True:#
            ef = self.e_f.forward(x, train)
            eg = self.e_g.forward(x, train)
            return ef + eg * u          
        else:
            return None
        
    def train(self, t, data):
        
        if t / self.dt >= self.time_th: # True: #
            if  not self.first_trained: # False:#
                opts = self.opts_pre
                self.first_trained = True
                epochs = 50
            else:
                opts = self.opts_post
                epochs = 1
                
            sample = data.get_D(t)
            
            for epoch in range(epochs):
                #running_loss = 0
                for x_i, k_i, dhe_real_i in zip(sample.x, sample.k, sample.dhe_real):
                    S_i = self.forward(x_i, k_i, t, train=True)
                    loss = F.mse_loss(torch.tensor(dhe_real_i),S_i)
                    self.e_f.model.zero_grad()
                    self.e_g.model.zero_grad()
                    
                    #running_loss += loss.item()

                    loss.backward()
                    opts['e_f'].step()
                    opts['e_g'].step()
                    
                #print(running_loss)
