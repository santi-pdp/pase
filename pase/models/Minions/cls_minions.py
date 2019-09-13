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

    elif cfg["name"] == "gap":
        return Gap(cfg, emb_dim)

    else:
        return minion_maker(cfg)

def make_samples(x,augment):

        x_pos = torch.cat((x[0], x[1]), dim=1)
        x_neg = torch.cat((x[0], x[2]), dim=1)

        if augment:
            x_pos2 = torch.cat((x[1], x[0]), dim=1)
            x_neg2 = torch.cat((x[1], x[2]), dim=1)

            x_pos=torch.cat((x_pos,x_pos2),dim=0)
            x_neg=torch.cat((x_neg,x_neg2),dim=0)



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

        if 'augment' in cfg.keys():
            self.augment=cfg['augment']
        else:
            self.augment=False

        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        x_pos, x_neg = make_samples(x,self.augment)
        x = torch.cat((x_pos, x_neg), dim=0).to(device)
        y = self.minion(x, alpha)
        label = make_labels(y).to(device)
        return y, label

class GIM(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])

        cfg['num_inputs'] = 2 * emb_dim

        if 'augment' in cfg.keys():
            self.augment=cfg['augment']
        else:
            self.augment=False


        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        x_pos, x_neg = make_samples(x,self.augment)
        x = torch.cat((x_pos, x_neg), dim=0).to(device)
        x = torch.mean(x, dim=2, keepdim=True)
        y = self.minion(x, alpha)
        label = make_labels(y).to(device)
        return y, label

class SPC(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])
        cfg['num_inputs'] = emb_dim

        self.minion = minion_maker(cfg)

        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        y = self.minion(x, alpha)
        label = make_labels(y).to(device)
        return  y, label

class Gap(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])

        cfg['num_inputs'] = 2 * emb_dim

        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        y, label = self.minion(x, alpha)
        label = label.float().to(device)
        return y, label

class AdversarialChunk(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])

        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        y, label = self.minion(x, alpha)
        label = label.float().to(device)
        return y, label
