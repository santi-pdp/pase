import torch
import torch.nn as nn
from ..frontend import WaveFe
from ..modules import *
from .minions import *
import torch.nn.functional as F
import json
import random

def cls_worker_maker(cfg, emb_dim):
    print("=" * 50)
    print("name", cfg["name"])
    print("=" * 50)
    if cfg["name"] == "mi":
        return LIM(cfg, emb_dim)

    elif cfg["name"] == "cmi":
        return GIM(cfg, emb_dim)

    elif cfg["name"] == "spc":
        return SPC(cfg, emb_dim)

    else:
        raise TypeError('Unrecognized minion type {}'.format(cfg["name"]))

    return Regress_minion

def make_samples(x):
    x_pos = torch.cat((x[0], x[1]), dim=1)
    x_neg = torch.cat((x[0], x[2]), dim=1)
    return  x_pos, x_neg


def make_labels(y):
    bsz = y.size(0) // 2
    slen = y.size(2)
    label = torch.cat((torch.ones(bsz, 1, slen, requires_grad=False), torch.zeros(bsz, 1, slen, requires_grad=False)), dim=0)
    return label

class LIM(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])

        cfg['num_inputs'] = 2 * emb_dim

        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss

    def forward(self, x, device):
        x_pos, x_neg = make_samples(x)
        x = torch.cat((x_pos, x_neg), dim=0).to(device)
        y = self.minion(x)
        label = make_labels(y).to(device)
        return y, label

class GIM(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])

        cfg['num_inputs'] = 2 * emb_dim

        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss

    def forward(self, x, device):
        x_pos, x_neg = make_samples(x)
        x = torch.cat((x_pos, x_neg), dim=0).to(device)
        x = torch.mean(x, dim=2, keepdim=True)
        y = self.minion(x)
        label = make_labels(y).to(device)
        return y, label

class SPC(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])
        cfg['num_inputs'] = emb_dim

        self.minion = minion_maker(cfg)

        self.loss = self.minion.loss

    def forward(self, x, device):
        y = self.minion(x)
        label = make_labels(y).to(device)
        return  y, label

