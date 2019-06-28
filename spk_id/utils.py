import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim.lr_scheduler as lr_scheduler
from ahoproc_tools.io import *
import os
import pickle
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import librosa


def build_valid_list(tr_list, spk2idx, va_split=0.2):
    if va_split > 0:
        # apply a split of x% to each speaker in the tr_list
        spk2utts = {}
        for tr_file in tr_list:
            spk = spk2idx[tr_file]
            if spk not in spk2utts:
                spk2utts[spk] = []
            spk2utts[spk].append(tr_file)
        va_files = []
        tr_files = []
        # Now select a split amount per speaker and rebuild train and valid lists
        for spk, files in spk2utts.items():
            spk_N = len(files)
            shuffle(files)
            spk_vaN = int(np.floor(spk_N * va_split))
            va_files += files[:spk_vaN]
            tr_files += files[spk_vaN:]
        return tr_files, va_files
    else:
        return tr_files, []

def compute_utterances_durs(files, data_root):
    durs = []
    for file_ in files:
        wav, rate = librosa.load(os.path.join(data_root,
                                              file_),
                                 sr=None)
        durs.append(wav.shape[0])
    return durs, rate

def compute_aco_durs(files, data_root, order=39,
                     ext='mfcc', np_fmt=False):
    durs = []
    for file_ in files:
        bname = os.path.splitext(file_)[0]
        if np_fmt:
            data = np.load(os.path.join(data_root, 
                                        bname + '.' + ext))
        else:
            data = read_aco_file(os.path.join(data_root, 
                                              bname + '.' + ext), 
                                 (-1, order))
        durs.append(data.shape[0])
    return durs

class LibriSpkIDMFCCDataset(Dataset):
    
    def __init__(self, data_root, files_list, spk2idx, order,
                 stats_f=None, ext='mfcc', np_fmt=False):
        super().__init__()
        self.files_list = files_list
        self.data_root = data_root
        self.spk2idx = spk2idx
        self.order = order
        self.ext = ext
        self.np_fmt = np_fmt
        with open(stats_f, 'rb') as f:
            self.stats = pickle.load(f)

    def __getitem__(self, idx):
        fpath = os.path.join(self.data_root, self.files_list[idx])
        bname = os.path.splitext(fpath)[0]
        #data = np.load(bname + '.npy')
        if self.np_fmt:
            data = np.load(bname + '.' + self.ext)
        else:
            data = read_aco_file(bname + '.' + self.ext, 
                                 (-1, self.order))
        data = data - np.array(self.stats['mean'])
        data = data / np.array(self.stats['std'])
        lab = self.spk2idx[self.files_list[idx]]
        return data, lab

    def __len__(self):
        return len(self.files_list)

class Collater(object):
    
    def __init__(self, max_len=None):
        self.max_len = max_len
        
    def __call__(self, batch):
        if self.max_len is None:
            # track max seq len in batch
            # and apply it padding others seqs
            max_len = 0
            for sample in batch:
                mfcc, lab = sample
                clen = mfcc.shape[0]
                if clen > max_len:
                    max_len = clen
        else:
            max_len = self.max_len
        X = []
        Y = []
        slens = []
        for sample in batch:
            mfcc, lab = sample
            clen = mfcc.shape[0]
            if clen < max_len:
                # pad with zeros in the end
                P = max_len - clen
                pad = np.zeros((P, mfcc.shape[1]))
                mfcc = np.concatenate((mfcc, pad), axis=0)
            elif clen > max_len:
                # trim the end (applied if we specify max_len externally)
                idxs = list(range(mfcc.shape[0] - max_len))
                bidx = random.choice(idxs)
                mfcc = mfcc[bidx:bidx + max_len]
            X.append(mfcc)
            Y.append(lab)
            slens.append(clen)
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)
        slens = torch.LongTensor(slens)
        return X, Y, slens

def load_spk2idx(spk2idx_file):
    spk2idx = np.load(spk2idx_file)
    spk2idx = dict(spk2idx.any())
    return spk2idx

def accuracy(Y_, Y):
    # Get rid of temporal resolution here,
    # average likelihood in time and then
    # compute argmax and accuracy
    Y__avg = torch.mean(Y_, 2)
    pred = Y__avg.max(1, keepdim=True)[1]
    acc = pred.eq(Y[:, 0].view_as(pred)).float().mean()
    return acc

def select_optimizer(opts, model):
    if opts.opt == 'adam':
        return optim.Adam(model.parameters(),
                          opts.lr)
    elif opts.opt == 'sgd':
        return optim.SGD(model.parameters(),
                         opts.lr, momentum=opts.momentum)
    elif opts.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(),
                             opts.lr, alpha=0.95)
    else:
        raise TypeError('Unrecognized optimizer {}'.format(opts.opt))

def select_scheduler(opts, opt):
    if opts.sched_mode == 'plateau':
        sched = lr_scheduler.ReduceLROnPlateau(opt,
                                               mode=opts.plateau_mode,
                                               factor=opts.lrdec,
                                               patience=opts.patience,
                                               verbose=True)
    elif opts.sched_mode == 'step':
        sched = lr_scheduler.StepLR(opt, 
                                    step_size=opts.sched_step_size,
                                    gamma=opts.lrdec)
    else:
        raise TypeError('Unrecognized optimizer LR scheduler'
                        ' {}'.format(opts.sched_mode))
    return sched
