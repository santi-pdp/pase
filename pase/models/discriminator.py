import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
import numpy as np
import json
import os
try:
    from modules import *
except ImportError:
    from .modules import *


class WaveDiscriminator(nn.Module):

    def __init__(self, ninputs=1,
                 fmaps=[128, 128, 256, 256, 512, 100],
                 strides=[10, 4, 4, 1, 1, 1],
                 kwidths=[30, 30, 30, 3, 3, 3],
                 norm_type='snorm'
                ):
        super().__init__()
        self.aco_decimator = nn.ModuleList()
        ninp = ninputs
        for fmap, kwidth, stride in zip(fmaps, kwidths, strides):
            self.aco_decimator.append(GConv1DBlock(ninp, fmap, kwidth,
                                                   stride, norm_type=norm_type))
                                                   
            ninp = fmap
        self.out_fc = nn.Conv1d(fmaps[-1], 1, 1)
        if norm_type == 'snorm':
            nn.utils.spectral_norm(self.out_fc)
        self.norm_type = norm_type

    def build_conditionW(self, cond):
        if cond is not None:
            cond_dim = cond.size(1)
            if not hasattr(self, 'proj_W'):
                self.proj_W = nn.Linear(cond_dim, cond_dim, bias=False)
                if self.norm_type == 'snorm':
                    nn.utils.spectral_norm(self.proj_W)
                if cond.is_cuda:
                    self.proj_W.cuda()
        
    def forward(self, x, cond=None):
        self.build_conditionW(cond)
        h = x
        for di in range(len(self.aco_decimator)):
            dec_layer = self.aco_decimator[di]
            h = dec_layer(h)
        bsz, nfeats, slen = h.size()
        if cond is not None:
            cond = torch.mean(cond, dim=2)
            # project conditioner with bilinear W
            cond = self.proj_W(cond)
            h = torch.mean(h, dim=2)
            h = h.view(-1, nfeats)
            cond = cond.view(-1, nfeats)
            cls = torch.bmm(h.unsqueeze(1),
                            cond.unsqueeze(2)).squeeze(2)
            cls = cls.view(bsz, 1)
        y = self.out_fc(h.unsqueeze(2)).squeeze(2)
        y = y + cls
        return y.squeeze(1)

if __name__ == '__main__':
    waveD = WaveDiscriminator()
    x = torch.randn(1, 1, 16000)
    h = torch.randn(1, 100, 100)
    y = waveD(x, h)
    print('y size: ', y.size())
