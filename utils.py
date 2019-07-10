import json
import shlex
import subprocess
import random
import torch
import torch.nn as nn
from .losses import *
import random
from random import shuffle
from pase.models.discriminator import *
import torch.optim as optim


def pase_parser(cfg_fname, batch_acum=1, device='cpu', do_losses=True,
                frontend=None):
    with open(cfg_fname, 'r') as cfg_f:
        cfg_all = json.load(cfg_f)
        if do_losses:
            # change loss section
            for i, cfg in enumerate(cfg_all):
                loss_name = cfg_all[i]['loss']
                if hasattr(nn, loss_name):
                    # loss in nn Modules
                    cfg_all[i]['loss'] = getattr(nn, loss_name)()
                else:
                    if loss_name == 'LSGAN' or loss_name == 'GAN':
                        dnet_cfg = {}
                        if 'DNet_cfg' in cfg_all[i]:
                            dnet_cfg = cfg_all[i].pop('DNet_cfg')
                        dnet_cfg['frontend'] = frontend
                        # make DNet
                        DNet =  RNNDiscriminator(**dnet_cfg)
                        if 'Dopt_cfg' in cfg_all[i]:
                            Dopt_cfg = cfg_all[i].pop('Dopt_cfg')
                            Dopt = optim.RMSprop(DNet.parameters(),
                                                 Dopt_cfg['lr'])
                        else:
                            Dopt = optim.RMSprop(DNet.parameters(), 0.0005)
                    Dloss = 'L2' if loss_name == 'LSGAN' else 'BCE'
                    cfg_all[i]['loss'] = WaveAdversarialLoss(DNet, Dopt,
                                                             loss=Dloss,
                                                             batch_acum=batch_acum,
                                                             device=device)
        return cfg_all


def build_optimizer(opt_cfg, params):
    if isinstance(opt_cfg, str):
        with open(opt_cfg, 'r') as cfg_f:
            opt_cfg = json.load(cfg_f)
    opt_name = opt_cfg.pop('name')
    if 'sched' in opt_cfg:
        sched_cfg = opt_cfg.pop('sched')
    else:
        sched_cfg = None
    opt_cfg['params'] = params
    opt = getattr(optim, opt_name)(**opt_cfg)
    if sched_cfg is not None:
        sname = sched_cfg.pop('name')
        sched_cfg['optimizer'] = opt
        sched = getattr(optim.lr_scheduler, sname)(**sched_cfg)
        return opt, sched
    else:
        return opt, None

def chunk_batch_seq(X, seq_range=[90, 1000]):
    bsz, nfeats, slen = X.size()
    min_seq = seq_range[0]
    max_seq = min(slen, seq_range[1])
    # sample a random chunk size
    chsz = random.choice(list(range(min_seq, max_seq)))
    idxs = list(range(slen - chsz))
    beg_i = random.choice(idxs)
    return X[:, :, beg_i:beg_i + chsz]

def kfold_data(data_list, utt2class, folds=10, valid_p=0.1):
    # returns the K lists of lists, so each k-th component
    # is composed of 3 sub-lists
    #idxs = list(range(len(data_list)))
    # shuffle the idxs first
    #shuffle(idxs)
    # group by class first
    classes = set(utt2class.values())
    items = dict((k, []) for k in classes)
    for data_el in data_list:
        items[utt2class[data_el]].append(data_el)
    lens = {}
    test_splits = {}
    for k in items.keys():
        shuffle(items[k])
        lens[k] = len(items[k])
        TEST_SPLIT_K = int((1. / folds) * lens[k])
        test_splits[k] = TEST_SPLIT_K
    lists = []
    beg_i = dict((k, 0) for k in test_splits.keys())
    # now slide a window per fold
    for fi in range(folds):
        test_split = []
        train_split = []
        valid_split = []
        print('-' * 30)
        print('Fold {} splits:'.format(fi))
        for k, data in items.items():
            te_split = data[beg_i[k]:beg_i[k] + test_splits[k]]
            test_split += te_split
            tr_split = data[:beg_i[k]] + data[beg_i[k] + test_splits[k]:]
            # select train and valid splits
            tr_split = tr_split[int(valid_p * len(tr_split)):]
            va_split = tr_split[:int(valid_p * len(tr_split))]
            train_split += tr_split
            valid_split += va_split
            print('Split {} train: {}, valid: {}, test: {}'
                  ''.format(k, len(tr_split), len(va_split), len(te_split)))
        # build valid split within train_split
        lists.append([train_split, valid_split, test_split])
    return lists

class AuxiliarSuperviser(object):

    def __init__(self, cmd_file, save_path='.'):
        self.cmd_file = cmd_file
        with open(cmd_file, 'r') as cmd_f:
            self.cmd = [l.rstrip() for l in cmd_f]
        self.save_path = save_path

    def __call__(self, iteration, ckpt_path, cfg_path):
        assert isinstance(iteration, int)
        assert isinstance(ckpt_path, str)
        assert isinstance(cfg_path, str)
        for cmd in self.cmd:
            sub_cmd = cmd.replace('$model', ckpt_path)
            sub_cmd = sub_cmd.replace('$iteration', str(iteration))
            sub_cmd = sub_cmd.replace('$cfg', cfg_path)
            sub_cmd = sub_cmd.replace('$save_path', self.save_path)
            print('Executing async command: ', sub_cmd)
            #shsub = shlex.split(sub_cmd)
            #print(shsub)
            p = subprocess.Popen(sub_cmd,
                                shell=True)

