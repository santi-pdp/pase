import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import *


class WaveFe(Model):
    """ Convolutional front-end to process waveforms
        into a decimated intermediate representation 
    """
    def __init__(self, num_inputs=1, 
                 sincnet=True,
                 kwidths=[251, 10, 5, 5, 5, 5, 5, 5], 
                 #strides=[1, 5, 2, 1, 2, 1, 2, 2], 
                 strides=[1, 10, 2, 1, 2, 1, 2, 2], 
                 fmaps=[64, 64, 128, 128, 256, 256, 512, 512],
                 norm_type='bnorm',
                 pad_mode='reflect', sr=16000,
                 emb_dim=256,
                 name='WaveFe'):
        super().__init__(name=name) 
        # apply sincnet at first layer
        self.sincnet = sincnet
        self.kwidths = kwidths
        self.strides = strides
        self.fmaps = fmaps
        self.blocks = nn.ModuleList()
        assert len(kwidths) == len(strides)
        assert len(strides) == len(fmaps)
        ninp = num_inputs
        for n, (kwidth, stride, fmap) in enumerate(zip(kwidths, strides,
                                                       fmaps), start=1):
            if n > 1:
                # make sure sincnet is deactivated after first layer
                sincnet = False
            self.blocks.append(FeBlock(ninp, fmap, kwidth, stride,
                                       pad_mode=pad_mode,
                                       norm_type=norm_type,
                                       sincnet=sincnet,
                                       sr=sr))
            ninp = fmap
        # last projection
        self.W = nn.Conv1d(fmap, emb_dim, 1)
        self.emb_dim = emb_dim

    def forward(self, x):
        h = x
        for n, block in enumerate(self.blocks):
            h = block(h)
        y = self.W(h)
        return y

if __name__ == '__main__':
    wavefe = WaveFe(norm_type='bnorm')
    print(wavefe)
    wavefe.describe_params()
    x = torch.randn(1, 1, 16000)
    y = wavefe(x)
    print(y.size())
