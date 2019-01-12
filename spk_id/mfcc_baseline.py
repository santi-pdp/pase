import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import json
import glob
from utils import *
from tensorboardX import SummaryWriter
import random
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from ahoproc_tools.io import read_aco_file
import os
from waveminionet.models.frontend import WaveFe
from waveminionet.models.modules import Model
import librosa
from random import shuffle
import argparse


# Make Linear classifier model
class LinearClassifier(Model):
    
    def __init__(self, num_inputs=None,
                 num_spks=None, 
                 name='CLS'):
        super().__init__(name=name)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        self.fc = nn.Conv1d(num_inputs, num_spks, 1)
        self.act = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        h = self.fc(x)
        y = self.act(h)
        return y

class MLPClassifier(Model):
    
    def __init__(self, num_inputs=None,
                 num_spks=None, 
                 hidden_size=2048,
                 name='MLP'):
        super().__init__(name=name)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        self.model = nn.Sequential(
            nn.Conv1d(num_inputs, hidden_size, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, num_spks, 1),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.model(x)

def select_model(opts, num_spks):
    if opts.model == 'cls':
        model = LinearClassifier(num_inputs=opts.order, 
                                 num_spks=num_spks)
    elif opts.model == 'mlp':
        model = MLPClassifier(num_inputs=opts.order, num_spks=num_spks,
                              hidden_size=opts.hidden_size)
    else:
        raise TypeError('Unrecognized model {}'.format(opts.model))
    return model

def main(opts):
    CUDA = torch.cuda.is_available() and not opts.no_cuda
    device = 'cuda' if CUDA else 'cpu'
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(opts.seed)
    spk2idx = load_spk2idx(opts.spk2idx)
    NSPK=len(set(spk2idx.values()))

    # Open up guia and split valid
    with open(opts.train_guia) as tr_guia_f: 
        tr_files = [l.rstrip() for l in tr_guia_f]

    with open(opts.test_guia) as te_guia_f: 
        te_files = [l.rstrip() for l in te_guia_f]

    tr_files_, va_files = build_valid_list(tr_files, spk2idx,
                                           va_split=opts.va_split)
    # Build Datasets
    dset = LibriSpkIDMFCCDataset(opts.data_root,
                                 tr_files_, spk2idx,
                                 opts.order,
                                 opts.stats)
    va_dset = LibriSpkIDMFCCDataset(opts.data_root,
                                    va_files, spk2idx,
                                    opts.order, 
                                    opts.stats)
    te_dset = LibriSpkIDMFCCDataset(opts.data_root,
                                    te_files, spk2idx,
                                    opts.order, 
                                    opts.stats)
    cc = Collater(max_len=opts.max_len)
    dloader = DataLoader(dset, batch_size=opts.batch_size, collate_fn=cc,
                         shuffle=True)
    va_dloader = DataLoader(va_dset, batch_size=opts.batch_size, collate_fn=cc,
                            shuffle=False)
    te_dloader = DataLoader(te_dset, batch_size=opts.batch_size, collate_fn=cc,
                            shuffle=False)
    # Build Model
    #cls = LinearClassifier(num_inputs=opts.order, num_spks=NSPK)
    cls = select_model(opts, num_spks=NSPK)
    cls.to(device)
    # Build optimizer and scheduler
    opt = select_optimizer(opts, cls)
    #opt = optim.SGD(cls.parameters(), lr=opts.lr, momentum=opts.momentum)
    sched = lr_scheduler.ReduceLROnPlateau(opt,
        mode='max',
        factor=opts.lrdec,
        patience=opts.patience,
        verbose=True)
    # Make writer
    writer = SummaryWriter(opts.save_path)
    best_val_acc = 0
    # flag for saver
    best_val = False
    for epoch in range(1, opts.epoch + 1):
        train_epoch(dloader, cls, opt, epoch, opts.log_freq, writer=writer,
                    device=device)
        eloss, eacc = eval_epoch(va_dloader, cls, epoch, opts.log_freq,
                                 writer=writer, device=device, key='valid')
        sched.step(eacc)
        if eacc > best_val_acc:
            print('New best val acc: {:.3f} => {:.3f}. Patience: {}'
                  ''.format(best_val_acc, eacc, opts.patience))
            best_val_acc = eacc
            best_val = True

        cls.save(opts.save_path, epoch, best_val=best_val)
        best_val = False
        # Eval test on the fly with training/valid
        teloss, teacc = eval_epoch(te_dloader, cls, epoch, opts.log_freq,
                                   writer=writer, device=device, key='test')


def train_epoch(dloader_, model, opt, epoch, log_freq=1, writer=None,
                device='cpu'):
    model.train()
    global_idx = epoch * len(dloader_)
    timings = []
    beg_t = timeit.default_timer()
    for bidx, batch in enumerate(dloader_, start=1):
        opt.zero_grad()
        X, Y, slens = batch
        X = X.transpose(1, 2)
        X = X.to(device)
        Y = Y.to(device)
        Y_ = model(X)
        Y = Y.view(-1, 1).repeat(1, Y_.size(2))
        loss = F.nll_loss(Y_.squeeze(-1), Y)
        loss.backward()
        opt.step()
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        if bidx % log_freq == 0 or bidx >= len(dloader_):
            acc = accuracy(Y_, Y)
            log_str = 'Batch {:5d}/{:5d} (Epoch {:3d}, Gidx {:5d})' \
                      ' '.format(bidx, len(dloader_),
                                 epoch, global_idx)
            log_str += 'loss: {:.3f} '.format(loss.item())
            log_str += 'bacc: {:.2f} '.format(acc)
            log_str += 'mbtime: {:.3f} s'.format(np.mean(timings))
            print(log_str)
            if writer is not None:
                writer.add_scalar('train/loss', loss.item(),
                                  global_idx)
                writer.add_scalar('train/bacc', acc, global_idx)
        global_idx += 1

def eval_epoch(dloader_, model, epoch, log_freq=1, writer=None, device='cpu',
               key='eval'):
    model.eval()
    with torch.no_grad():
        eval_losses = []
        eval_accs = []
        timings = []
        beg_t = timeit.default_timer()
        for bidx, batch in enumerate(dloader_, start=1):
            X, Y, slens = batch
            X = X.transpose(1, 2)
            X = X.to(device)
            Y = Y.to(device)
            Y_ = model(X)
            Y = Y.view(-1, 1).repeat(1, Y_.size(2))
            loss = F.nll_loss(Y_, Y)
            eval_losses.append(loss.item())
            acc = accuracy(Y_, Y)
            eval_accs.append(acc)
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if bidx % log_freq == 0 or bidx >= len(dloader_):
                
                log_str = 'EVAL::{} Batch {:4d}/{:4d} (Epoch {:3d})' \
                          ' '.format(key, bidx, len(dloader_),
                                     epoch)
                log_str += 'loss: {:.3f} '.format(loss.item())
                log_str += 'bacc: {:.2f} '.format(acc)
                log_str += 'mbtime: {:.3f} s'.format(np.mean(timings))
                print(log_str)
        mloss = np.mean(eval_losses)
        macc = np.mean(eval_accs)
        if writer is not None:
            writer.add_scalar('{}/loss'.format(key), mloss,
                              epoch)
            writer.add_scalar('{}/acc'.format(key), macc, epoch)
        print('EVAL epoch {:3d} mean loss: {:.3f}, mean acc: {:.2f} '
             ''.format(epoch, mloss, macc))
        return mloss, macc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='ckpt_mfcc')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--test_guia', type=str, default=None)
    parser.add_argument('--train_guia', type=str, default=None)
    parser.add_argument('--spk2idx', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--va_split', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--order', type=int, default=39)
    parser.add_argument('--lrdec', type=float, default=0.1,
                        help='Decay factor of learning rate after '
                             'patience epochs of valid accuracy not '
                             'improving (Def: 0.1).')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--stats', type=str, default=None)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--model', type=str, default='cls',
                        help='(1) cls, (2) mlp (Def: cls).')
    
    opts = parser.parse_args()

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    main(opts)
