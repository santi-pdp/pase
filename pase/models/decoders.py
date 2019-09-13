import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .frontend import *
#from .Minions.minions import *
import random

class SpectrumLM(nn.Module):
    """ RNN lang model for spectrum frame preds """
    def __init__(self, rnn_size, rnn_layers, out_dim,
                 dropout,
                 cuda, rnn_type='LSTM',
                 bidirectional=False):
        super().__init__()

        self.do_cuda = cuda
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.out_dim = out_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        if bidirectional:
            self.dirs = 2
        else:
            self.dirs = 1
        assert rnn_type == 'LSTM' or rnn_type == 'GRU', rnn_type
        self.rnn = getattr(nn, rnn_type)(self.out_dim, self.rnn_size,
                                        self.rnn_layers,
                                        batch_first=True,
                                        dropout=self.dropout,
                                        bidirectional=bidirectional)
        self.out_fc = nn.Linear(self.rnn_size, self.out_dim)

    def forward(self, x, dec_steps, state=None, 
                dec_cps={}):
        # x is just a time-step input [B, F]
        assert len(x.size()) == 2, x.size()
        if state is None:
            state = self.init_hidden(x.size(0))
        assert isinstance(dec_cps, dict), type(dec_cps)
        x = x.unsqueeze(1)
        ht = x
        frames = []
        # forward through RNN
        for t in range(dec_steps):
            if t in dec_cps:
                #print('Using cp at t:{}'.format(t))
                ht = dec_cps[t]
                if len(ht.size()) == 2:
                    # add time axis
                    ht = ht.unsqueeze(1)
            #print('Forwarding ht: ', ht.size())
            ht, state = self.rnn(ht, state)
            ht = self.out_fc(ht)
            frames.append(ht)
        frames = torch.cat(frames, 1)
        return frames, state

    def init_hidden(self, bsz):
        h0 = Variable(torch.randn(self.dirs * self.rnn_layers,
                                  bsz, self.rnn_size))
        if self.do_cuda:
            h0 = h0.cuda()
        if self.rnn_type == 'LSTM':
            c0 = h0.clone()
            return (h0, c0)
        else:
            return h0

class SpectrogramDecoder(Model):

    def __init__(self, num_inputs, nfft=1024, 
                 strides=[1, 1, 1],
                 kwidths=[3, 3, 3],
                 fmaps=[256, 256, 256],
                 norm_type=None,
                 name='SpectrogramDecoder'):
        super().__init__(name=name)
        ninp = num_inputs
        self.dec = nn.ModuleList()
        for di, (kwidth, stride, fmap) in enumerate(zip(kwidths, strides,
                                                        fmaps), 
                                                    start=1):
            if stride > 1:
                self.dec.append(GDeconv1DBlock(ninp, fmap, kwidth, stride,
                                               norm_type=norm_type))
            else:
                self.dec.append(GConv1DBlock(ninp, fmap, kwidth, 1,
                                             norm_type=norm_type))
            ninp = fmap
        self.dec.append(nn.Conv1d(ninp, nfft // 2 + 1, 1))

    def forward(self, x):
        for dec in self.dec: x = dec(x)
        return x


if __name__ == '__main__':
    spec_dec = SpectrogramDecoder(1024, 1024)
    x = torch.randn(1, 512, 100)
    y = spec_dec(x)
    print(x.shape)
    print(y.shape)
