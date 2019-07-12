import torch
import torch.nn as nn
from .frontend import WaveFe
from .modules import *
import torch.nn.functional as F
import json
import random
from random import shuffle


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

    def __init__(self, ninp, fmaps, dout=0, bias=True,
                 kwidth=1,
                 norm_type=None,
                 name='MLPBlock'):
        super().__init__(name=name)
        self.ninp = ninp
        self.fmaps = fmaps
        self.kwidth = kwidth
        self.W = nn.Conv1d(ninp, fmaps, kwidth, bias=bias)
        self.norm = build_norm_layer(norm_type, self.W,
                                     fmaps)
        self.act = nn.PReLU(fmaps)
        self.dout = nn.Dropout(dout)
    
    def forward(self, x):
        if self.kwidth > 1:
            if self.kwidth % 2 == 0:
                x = F.pad(x, (self.kwidth // 2 - 1,
                              self.kwidth // 2))
            else:
                x = F.pad(x, (self.kwidth // 2, self.kwidth // 2))
        h = self.W(x)
        h = forward_norm(h, self.norm)
        h = self.act(h)
        return self.dout(h)

class DecoderMinion(Model):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 fmaps=[256, 256, 128, 128, 128, 64, 64],
                 strides=[2, 2, 2, 2, 2, 5],
                 kwidths=[2, 2, 2, 2, 2, 5],
                 norm_type=None,
                 out_tanh=True,
                 rnn_layers=0,
                 rnn_size=512,
                 rnn_type='qrnn',
                 shuffle_p=0,
                 bias=True,
                 detach_frontend=False,
                 skip=False,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 name='DecoderMinion'):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.skip = skip
        self.shuffle_p = shuffle_p
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.fmaps = fmaps
        self.strides = strides
        self.kwidths = kwidths
        self.detach_frontend = detach_frontend
        self.norm_type = norm_type
        self.loss = loss
        self.out_tanh = out_tanh
        self.loss_weight = loss_weight
        self.keys = keys
        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        # First go through deconvolving structure
        for (fmap, kw, stride) in zip(fmaps, kwidths, strides):
            block = GDeconv1DBlock(ninp, fmap, kw, stride,
                                   norm_type=norm_type,
                                   bias=bias)
            self.blocks.append(block)
            ninp = fmap

        for _ in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size, dropout,
                                        bias=bias, norm_type=norm_type))
            ninp = hidden_size
        if rnn_layers > 0:
            self.rnn_block = build_rnn_block(ninp, rnn_size,
                                             rnn_layers, rnn_type,
                                             dropout=dropout)
            ninp = rnn_size
        self.W = nn.Conv1d(ninp, num_outputs, 1, bias=bias)
        
    def forward(self, x):
        if self.detach_frontend:
            x = x.detach()
        h = x
        if self.shuffle_p > 0:
            do_shuffle = random.random() <= self.shuffle_p
            if do_shuffle:
                h = list(torch.chunk(h, h.size(2), dim=2))
                shuffle(h)
                h = torch.cat(h, dim=2)
        for bi, block in enumerate(self.blocks, start=1):
            h_ = h
            h = block(h)
        y = self.W(h)
        if hasattr(self, 'rnn_block'):
            y, _ = self.rnn_block(y)
        if self.out_tanh:
            y = torch.tanh(y)
        if self.skip:
            return y, h
        else:
            return y
                 

class MLPMinion(Model):

    def __init__(self, num_inputs, 
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 shuffle_p=0,
                 norm_type=None,
                 skip=True,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 grad_reverse=False,
                 kwidths=[],
                 name='MLPMinion'):
        super().__init__(name=name)
        # Implemented with Conv1d layers to not 
        # transpose anything in time, such that
        # frontend and minions are attached very simply
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.norm_type = norm_type
        self.skip = skip
        self.shuffle_p = shuffle_p
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.grad_reverse = grad_reverse
        self.loss_weight = loss_weight
        self.keys = keys
        if len(kwidths) > 0:
            # assert each layer has a kwidth specified
            assert len(kwidths) == hidden_layers, len(kwidths)
        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        for lidx in range(hidden_layers):
            if len(kwidths) > 0:
                kw = kwidths[lidx]
            else:
                kw = 1
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size,
                                        dropout,
                                        kwidth=kw,
                                        norm_type=norm_type))
            ninp = hidden_size
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        
    def forward(self, x):
        if self.grad_reverse:
            x = GradReverse.apply(x)
        h = x
        if self.shuffle_p > 0:
            do_shuffle = random.random() <= self.shuffle_p
            if do_shuffle:
                h = list(torch.chunk(h, h.size(2), dim=2))
                shuffle(h)
                h = torch.cat(h, dim=2)
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
                 shuffle_p=0,
                 skip=True,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 name='GRUMinion'):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.skip = skip
        self.shuffle_p = shuffle_p
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.loss_weight = loss_weight
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
        h = x
        if self.shuffle_p > 0:
            do_shuffle = random.random() <= self.shuffle_p
            if do_shuffle:
                h = list(torch.chunk(h, h.size(2), dim=2))
                shuffle(h)
                h = torch.cat(h, dim=2)
        h, _ = self.rnn(h.transpose(1, 2))
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
                 loss_weight=1.,
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
                         loss_weight=loss_weight,
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
