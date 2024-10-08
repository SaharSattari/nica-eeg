import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class tcl(nn.module):
    def __init__(self, input_dim, hidden_dim, num_class, num_layers, activation = 'maxout', pool_size = 2,
                 slope = 0.1, feature_nonlinearity = 'abs'):
        super(tcl, self).__init__()
        self.feature_nonlinearity = feature_nonlinearity

        #shared feature-MLP
        self.MLP = MLP(input_dim, hidden_dim, num_layers, activation = activation, pool_size = pool_size, slope = slope)
       
        #MLRs (subject specific mlr)
        num_sub = len(num_class)   # class is refering to number of distinct time windows

        if isinstance(hidden_dim, list):
            _mlr_input_dim = hidden_dim[-1]  # if the dimension of hidden layers differ (it is a list) then use the last element of this list
        else:
            _mlr_input_dim = hidden_dim

        _MLRs_list = [nn.Linear(_mlr_input_dim, num_class[k]) for k in range(num_sub)]
        self.MLRs = nn.ModuleList(_MLRs_list)

        for k in range(num_sub):
            torch.nn.init.xavier_uniform_(self.MLRs[k].weight)
        
            
    def forward(self, x, sub_id = None):
        """forward pass
        Args:
            x: shape(batch_size,num_channels)
            sub_id: subject id
        Returns:
            y: labels (batch_size,)
            h: features (batch_size, num_channels)
        """
        h = self.MLP(x)
        
        if self.feature_nonlinearity == 'abs':
            h = torch.abs(h) # Nonlinearity of the last hidden layer (feature value)
            
        # logis
        if sub_id is None:
            y = None
        else:
            uniq = torch.unique(sub_id)
            y = [self.MLRs[k](h[(sub_id == k).nonzero().squeeze(),:]) for k in uniq]
            y = torch.concatenate(y,axis=0)
            
        return y, h



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, 
                 num_layers, activation = 'Maxout', pool_size = 2,
                 slope = .1):
        """Built feature-MLP model as feature extractor:
        Args:
            input_dim: size of input channels, here is number of components
            hidden_dim: size of nodes for each layer
            num_layers: number of layers == len(hidden_dim)
            activation: (option) activation function in the middle layer
            pool_size: pool size of max-out nonlinearity
            slope: for ReLU and leaky_relu activation
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        # Checking hidden_dim
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * (self.num_layers) # if it is a single integer, make a list with the size of hidden layers and same number of neurons for each layer
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))
        
        # Activation
        if isinstance(activation, str):
            self.activation = [activation] * (self.num_layers - 1) # No activation in last layer
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))
        
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'ReLU':
                self._act_f.append(lambda x: F.relu(x))
            elif act == 'Maxout':
                self._act_f.append(Maxout(pool_size))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        # MLP
        if activation == 'Maxout':
            _layer_list = [nn.Linear(self.input_dim, self.hidden_dim[0]*pool_size)] # compensate maxout pool size
            for k in range(1,len(hidden_dim)-1):
                _layer_list.append(nn.Linear(self.hidden_dim[k - 1], self.hidden_dim[k]*pool_size)) # compensate maxout pool size
            _layer_list.append(nn.Linear(self.hidden_dim[-2], self.hidden_dim[-1])) # last layer is not a maxout
        else:
            _layer_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for k in range(1,len(hidden_dim)):
                _layer_list.append(nn.Linear(self.hidden_dim[k - 1], self.hidden_dim[k]))
            
        self.layers = nn.ModuleList(_layer_list)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.2)
        
        # Bacth-norm
        self.bn = nn.ModuleList()
        for bni in range(self.num_layers-1):
            self.bn.append(nn.BatchNorm1d(self.hidden_dim[bni]))
            
        # initialize
        for k in range(len(self.layers)):
            torch.nn.init.xavier_uniform_(self.layers[k].weight)
        
    def forward(self, x):
        """forward process
        Args:
            x: input data nput [batch, dim]
        Returns:
            h: features
        """
        #h/feature values
        h = x
        for k in range(len(self.layers)):
            if k == len(self.layers) - 1:
                h = self.layers[k](h)
            else:
                h = self._act_f[k](self.layers[k](h))
                if k<=1: h=self.dropout(h)

        return h
    
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        m, _ = torch.max(torch.reshape(x, (*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:])), dim=2)
        return m
