import torch
import torch.nn.functional as F
import torch.nn as nn
import json
if  __name__ == '__main__':
    from modules import *
else:
    from .modules import *


def wf_builder(cfg_path):
    with open(cfg_path, 'r') as cfg_f:
        cfg = json.load(cfg_f)
        return WaveFe(**cfg)

class WaveFe(Model):
    """ Convolutional front-end to process waveforms
        into a decimated intermediate representation 
    """
    def __init__(self, num_inputs=1, 
                 sincnet=True,
                 kwidths=[251, 10, 5, 5, 5, 5, 5, 5], 
                 strides=[1, 10, 2, 1, 2, 1, 2, 2], 
                 fmaps=[64, 64, 128, 128, 256, 256, 512, 512],
                 norm_type='bnorm',
                 pad_mode='reflect', sr=16000,
                 emb_dim=256,
                 rnn_pool=False,
                 vq_K=None,
                 vq_beta=0.25,
                 vq_gamma=0.99,
                 norm_out=False,
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
        if rnn_pool:
            self.rnn = nn.GRU(fmap, emb_dim // 2, bidirectional=True, 
                              batch_first=True)
            self.W = nn.Linear(emb_dim, emb_dim)
        else:
            self.W = nn.Conv1d(fmap, emb_dim, 1)
        self.emb_dim = emb_dim
        self.rnn_pool = rnn_pool
        if vq_K is not None and vq_K > 0:
            self.quantizer = VQEMA(vq_K, self.emb_dim,
                                   vq_beta, vq_gamma)
        else:
            self.quantizer = None
        # ouptut vectors are normalized to norm^2 1
        if norm_out:
            self.norm_out = nn.BatchNorm1d(self.emb_dim, affine=False)

        
    def forward(self, x):
        h = x
        for n, block in enumerate(self.blocks):
            h = block(h)
        if self.rnn_pool:
            ht, _ = self.rnn(h.transpose(1, 2))
            y = self.W(ht) 
            y = y.transpose(1, 2)
        else:
            y = self.W(h)
        if hasattr(self, 'norm_out'):
            y = self.norm_out(y)

        if self.quantizer is not None:
            qloss, y, pp, enc = self.quantizer(y)
            if self.training:
                return qloss, y, pp, enc
            else:
                return y
        return y

if __name__ == '__main__':
    from modules import *
    wavefe = WaveFe(norm_type='bnorm')
    print(wavefe)
    wavefe.describe_params()
    x = torch.randn(1, 1, 16000)
    y = wavefe(x)
    print(y.size())
    vq = VQEMA(50, 20, 0.25, 0.99)
    _, yq, _ , _ = vq(y)
    print(yq.size())
    qwavefe = WaveFe(norm_type='bnorm', emb_dim=20)
    qwavefe.eval()
    yq2 = qwavefe(x)
    print(yq2.size())
    # try builder
    wfb = wf_builder('../../cfg/frontend_RF160ms_emb100.cfg')
    print(wfb)
