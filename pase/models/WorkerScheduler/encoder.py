import torch.nn as nn
from ..modules import *
from ..aspp import ASPP, aspp_resblock
import torch.nn.functional as F
import json
import random

class encoder(Model):

    def __init__(self, frontend, name='encoder'):
        super().__init__(name)
        self.frontend = frontend
        self.emb_dim = self.frontend.emb_dim

    def forward(self, batch, device):

        if type(batch) == dict:
            x = torch.cat((batch['chunk'],
                                 batch['chunk_ctxt'],
                                 batch['chunk_rand']),
                                dim=0).to(device)
        else:
            x = batch

        y = self.frontend(x)

        if type(batch) == dict:
            embedding = torch.chunk(y, 3, dim=0)

            chunk = embedding[0]

            return embedding, chunk
        else:
            return y


class aspp_res_encoder(Model):

    def __init__(self, sinc_out, hidden_dim, kernel_sizes=[11, 11, 11, 11], strides=[10, 4, 2, 2], dilations=[1, 6, 12, 18], fmaps=48, name='aspp_encoder', pool2d=False, rnn_pool=False, dense=False):
        super().__init__(name=name)
        self.sinc = SincConv_fast(1, sinc_out, 251,
                                  sample_rate=16000,
                                  padding='SAME',
                                  stride=1,
                                  pad_mode='reflect'
                                  )


        self.ASPP_blocks = nn.ModuleList()

        for i in range(len(kernel_sizes)):
            if i == 0:
                self.ASPP_blocks.append(aspp_resblock(sinc_out, hidden_dim, kernel_sizes[i], strides[i], dilations, fmaps[i], pool2d[i], dense))
            else:
                self.ASPP_blocks.append(aspp_resblock(hidden_dim, hidden_dim, kernel_sizes[i], strides[i], dilations, fmaps[i], pool2d[i], dense))


        self.rnn_pool = rnn_pool

        if rnn_pool:
            self.rnn = build_rnn_block(hidden_dim, hidden_dim // 2,
                                       rnn_layers=1,
                                       rnn_type='qrnn',
                                       bidirectional=True,
                                       dropout=0)
            self.W = nn.Conv1d(hidden_dim, hidden_dim, 1)


        self.emb_dim = hidden_dim



    def forward(self, batch, device):

        if type(batch) == dict:
            x = torch.cat((batch['chunk'],
                           batch['chunk_ctxt'],
                           batch['chunk_rand']),
                          dim=0).to(device)
        else:
            x = batch

        sinc_out = self.sinc(x)

        out = sinc_out
        for block in self.ASPP_blocks:
            out = block(out)

        h = out

        if self.rnn_pool:
            h = h.transpose(1, 2).transpose(0, 1)
            h, _ = self.rnn(h)
            h = h.transpose(0, 1).transpose(1, 2)


        if type(batch) == dict:
            embedding = torch.chunk(h, 3, dim=0)

            chunk = embedding[0]
            return embedding, chunk
        else:
            return h

