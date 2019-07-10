import torch.nn as nn
from ..frontend import WaveFe
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

class aspp_encoder(Model):

    def __init__(self, sinc_out, hidden_dim):
        super().__init__(name='aspp_encoder')
        self.sinc = SincConv_fast(1, sinc_out, 251,
                                  sample_rate=16000,
                                  padding='SAME',
                                  stride=160,
                                  pad_mode='reflect'
                                  )

        self.block1 = nn.Sequential(ASPP(sinc_out, hidden_dim),
                                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=11, stride=1, padding=5,
                                              bias=False),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(hidden_dim))

        self.block2 = nn.Sequential(ASPP(hidden_dim, hidden_dim),
                                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=11, stride=1, padding=5,
                                              bias=False),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(hidden_dim))

        # self.fc = nn.Linear(hidden_dim, hidden_dim)

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

        out_1 = self.block1(sinc_out)

        out_2 = self.block2(out_1)

        y = out_1 + out_2


        if type(batch) == dict:
            embedding = torch.chunk(y, 3, dim=0)

            chunk = embedding[0]
            return embedding, chunk
        else:
            return y

class aspp_res_encoder(Model):

    def __init__(self, sinc_out, hidden_dim):
        super().__init__(name='aspp_encoder')
        self.sinc = SincConv_fast(1, sinc_out, 251,
                                  sample_rate=16000,
                                  padding='SAME',
                                  stride=160,
                                  pad_mode='reflect'
                                  )

        self.block1 = aspp_resblock(sinc_out, hidden_dim)

        self.block2 = aspp_resblock(hidden_dim, hidden_dim)

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

        out_1 = self.block1(sinc_out)

        out_2 = self.block2(out_1)

        y = out_1 + out_2


        if type(batch) == dict:
            embedding = torch.chunk(y, 3, dim=0)

            chunk = embedding[0]
            return embedding, chunk
        else:
            return y
