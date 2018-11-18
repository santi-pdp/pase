import torch
import torch.nn as nn
from .frontend import WaveFe
from .modules import *
import torch.nn.functional as F
import json


def minion_maker(cfg):
    mtype = cfg.pop('type', 'mlp')
    if mtype == 'mlp':
        minion = MLPMinion(**cfg)
    elif mtype == 'decoder':
        minion = DecoderMinion(**cfg)
    else:
        raise TypeError('Unrecognized minion type {}'.format(mtype))
    return minion

class MLPBlock(NeuralBlock):

    def __init__(self, ninp, fmaps, dout=0, name='MLPBlock'):
        super().__init__(name=name)
        self.ninp = ninp
        self.fmaps = fmaps
        self.W = nn.Conv1d(ninp, fmaps, 1)
        self.act = nn.PReLU(fmaps)
        self.dout = nn.Dropout(dout)
    
    def forward(self, x):
        return self.dout(self.act(self.W(x)))

class DecoderMinion(Model):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 fmaps=[256, 256, 128, 128, 64, 64],
                 strides=[2, 2, 2, 2, 5],
                 kwidths=[2, 2, 2, 2, 5],
                 norm_type=None,
                 skip=False,
                 loss=None,
                 keys=None,
                 name='DecoderMinion'):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.fmaps = fmaps
        self.strides = strides
        self.kwidths = kwidths
        self.norm_type = norm_type
        self.loss = loss
        self.keys = keys
        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        # First go through deconvolving structure
        for (fmap, kw, stride) in zip(fmaps, kwidths, strides):
            block = GDeconv1DBlock(ninp, fmap, kw, stride,
                                   norm_type=norm_type)
            self.blocks.append(block)
            ninp = fmap

        for _ in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size, dropout))
            ninp = hidden_size
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        
    def forward(self, x):
        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h_ = h
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y
                 
class MLPMinion(Model):

    def __init__(self, num_inputs, 
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 skip=True,
                 loss=None,
                 keys=None,
                 name='MLPMinion'):
        super().__init__(name=name)
        # Implemented with Conv1d layers to not 
        # transpose anything in time, such that
        # frontend and minions are attached very simply
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.keys = keys
        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        for _ in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size,
                                        dropout))
            ninp = hidden_size
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        
    def forward(self, x):
        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y

if __name__ == '__main__':
    #minion = MLPMinion(256, 128, 0)
    minion = DecoderMinion(256, 128, 0)
    x = torch.randn(1, 256, 200)
    print(x)
    minion.describe_params()
    y, h = minion(x)
    print(h.size())
    print(y.size())
