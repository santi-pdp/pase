import torch
import torch.nn as nn
from .frontend import WaveFe
from .modules import *
import torch.nn.functional as F
import json
import random


def minion_maker(cfg):
    mtype = cfg.pop('type', 'mlp')
    if mtype == 'mlp':
        minion = MLPMinion(**cfg)
    elif mtype == 'decoder':
        minion = DecoderMinion(**cfg)
    elif mtype == 'spc':
        minion = SPCMinion(**cfg)
    elif mtype == 'gru':
        minion = GRUMinion(**cfg)
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
                 fmaps=[256, 256, 128, 128, 128, 64, 64],
                 strides=[2, 2, 2, 2, 2, 5],
                 kwidths=[2, 2, 2, 2, 2, 5],
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

class GRUMinion(Model):

    def __init__(self, num_inputs, 
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 skip=True,
                 loss=None,
                 keys=None,
                 name='GRUMinion'):
        super().__init__(name=name)
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
        self.rnn = nn.GRU(ninp,
                          hidden_size,
                          num_layers=hidden_layers,
                          batch_first=True,
                          dropout=dropout)
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        
    def forward(self, x):
        h, _ = self.rnn(x.transpose(1, 2))
        h = h.transpose(1, 2)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y

class SPCMinion(MLPMinion):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 ctxt_frames=5,
                 seq_pad=16,
                 skip=True,
                 loss=None,
                 keys=None,
                 name='SPCMinion'):
        # num_inputs is code dimension in each time-step,
        # so the MLP has [num_inputs x ctxt_frames] inputs
        # as we unroll time dimension to fixed-sized windows
        print('num_inputs: ', num_inputs)
        print('ctxt_frames: ', ctxt_frames)
        num_inputs = (ctxt_frames + 1) * num_inputs
        print('num_inputs: ', num_inputs)
        super().__init__(num_inputs=num_inputs,
                         num_outputs=num_outputs,
                         dropout=dropout,
                         hidden_size=hidden_size,
                         hidden_layers=hidden_layers,
                         skip=skip,
                         loss=loss,
                         keys=keys,
                         name=name)
        self.ctxt_frames = ctxt_frames
        self.seq_pad = seq_pad

    def forward(self, x):
        # x is a batch of sequences
        # of dims [B, channels, time]
        # first select a "central" time-step
        # with enough seq_pad an ctxt_frames
        # margin M = seq_pad + ctxt_frames on both sides
        seq_pad = self.seq_pad
        N = self.ctxt_frames
        M = seq_pad + N
        idxs_t = list(range(M+1, x.size(2) - M))
        t = random.choice(idxs_t)

        bsz = x.size(0)

        # now select future_t (to begin future seq)
        idxs_ft = list(range(t + seq_pad, x.size(2) - N))
        future_t = random.choice(idxs_ft)
        idxs_pt = list(range(N, t - seq_pad))
        past_t = random.choice(idxs_pt)

        # chunk input sequences and current frame
        future = x[:, :, future_t:future_t + N].contiguous().view(bsz, -1)
        past = x[:, :, past_t- N:past_t].contiguous().view(bsz, -1)
        current = x[:, :, t].contiguous()
        
        # positive batch (future data)
        pos = torch.cat((current, future), dim=1)
        # negative batch (past data)
        neg = torch.cat((current, past), dim=1)

        # forward both jointly
        x_full = torch.cat((pos, neg), dim=0).unsqueeze(2)
        h = x_full
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
