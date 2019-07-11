import torch.nn as nn
from ..frontend import WaveFe
from ..modules import *
import torch.nn.functional as F
import json
import random

class encoder(Model):

    def __init__(self, frontend, name='encoder'):
        super().__init__(name)
        self.frontend = frontend
        self.emb_dim = self.frontend.emb_dim

    def forward(self, batch, device):
        x = torch.cat((batch['chunk'],
                             batch['chunk_ctxt'],
                             batch['chunk_rand']),
                            dim=0).to(device)

        y = self.frontend(x)

        embedding = torch.chunk(y, 3, dim=0)

        chunk = embedding[0]

        return embedding, chunk