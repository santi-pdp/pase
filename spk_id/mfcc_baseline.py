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
                 z_bnorm=False,
                 name='CLS'):
        super().__init__(name=name)
        if z_bnorm:
            # apply z-norm to the input
            self.z_bnorm = nn.BatchNorm1d(frontend.emb_dim, affine=False)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        self.fc = nn.Conv1d(num_inputs, num_spks, 1)
        self.act = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        h = x
        if hasattr(self, 'z_bnorm'):
            h = self.z_bnorm(h)
        h = self.fc(x)
        y = self.act(h)
        return y

class MLPClassifier(Model):
    
    def __init__(self, num_inputs=None,
                 num_spks=None, 
                 hidden_size=2048,
                 z_bnorm=False,
                 name='MLP'):
        super().__init__(name=name, max_ckpts=1000)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        if z_bnorm:
            # apply z-norm to the input
            self.z_bnorm = nn.BatchNorm1d(frontend.emb_dim, affine=False)
        self.model = nn.Sequential(
            nn.Conv1d(num_inputs, hidden_size, 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, num_spks, 1),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        h = x
        if hasattr(self, 'z_bnorm'):
            h = self.z_bnorm(h)
        return self.model(h)

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

    model = select_model(opts, num_spks=NSPK)
    model.to(device)
    print(model)
    if opts.train:
        print('=' * 20)
        print('Entering TRAIN mode')
        print('=' * 20)

        with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
            cfg_f.write(json.dumps(vars(opts), indent=2))

        # Open up guia and split valid
        with open(opts.train_guia) as tr_guia_f: 
            tr_files = [l.rstrip() for l in tr_guia_f]


        tr_files_, va_files = build_valid_list(tr_files, spk2idx,
                                               va_split=opts.va_split)
        # compute total samples dur
        beg_t = timeit.default_timer()
        tr_durs = compute_aco_durs(tr_files_, opts.data_root, opts.order, 
                                   ext=opts.ext, np_fmt=opts.np_fmt)
        va_durs = compute_aco_durs(va_files, opts.data_root, opts.order,
                                   ext=opts.ext, np_fmt=opts.np_fmt)
        train_dur = np.sum(tr_durs)
        valid_dur = np.sum(va_durs)
        hop = 160
        sr = 16000
        end_t = timeit.default_timer()
        print('Read tr/va {:.1f} s/{:.1f} s in {} s'.format((train_dur * hop) / sr,
                                                            (valid_dur * hop) / sr,
                                                            end_t - beg_t))
        # Build Datasets
        dset = LibriSpkIDMFCCDataset(opts.data_root,
                                     tr_files_, spk2idx,
                                     opts.order,
                                     opts.stats,
                                     ext=opts.ext,
                                     np_fmt=opts.np_fmt)
        va_dset = LibriSpkIDMFCCDataset(opts.data_root,
                                        va_files, spk2idx,
                                        opts.order, 
                                        opts.stats,
                                        ext=opts.ext,
                                        np_fmt=opts.np_fmt)
        cc = Collater(max_len=opts.max_len)
        dloader = DataLoader(dset, batch_size=opts.batch_size, collate_fn=cc,
                             shuffle=True)
        va_dloader = DataLoader(va_dset, batch_size=opts.batch_size, collate_fn=cc,
                                shuffle=False)
        tr_bpe = (train_dur // opts.max_len) // opts.batch_size
        va_bpe = (valid_dur // opts.max_len) // opts.batch_size

        te_dloader = None
        # Build optimizer and scheduler
        opt = select_optimizer(opts, model)
        sched = select_scheduler(opts, opt)
        # Make writer
        writer = SummaryWriter(opts.save_path)
        best_val_acc = 0
        # flag for saver
        best_val = False
        for epoch in range(1, opts.epoch + 1):
            train_epoch(dloader, model, opt, epoch, opts.log_freq, writer=writer,
                        device=device, bpe=tr_bpe)
            eloss, eacc = eval_epoch(va_dloader, model, epoch, opts.log_freq,
                                     writer=writer, device=device, bpe=va_bpe,
                                     key='valid')
            if opts.sched_mode == 'step':
                sched.step()
            else:
                sched.step(eacc)
            if eacc > best_val_acc:
                print('*' * 40)
                print('New best val acc: {:.3f} => {:.3f}.'
                      ''.format(best_val_acc, eacc))
                print('*' * 40)
                best_val_acc = eacc
                best_val = True
            model.save(opts.save_path, epoch - 1, best_val=best_val)
            best_val = False
    if opts.test:
        print('=' * 20)
        print('Entering TEST mode')
        print('=' * 20)

        model.load_pretrained(opts.test_ckpt, load_last=True, verbose=True)
        model.to(device)
        model.eval()
        with open(opts.test_guia) as te_guia_f: 
            te_files = [l.rstrip() for l in te_guia_f]
            te_dset = LibriSpkIDMFCCDataset(opts.data_root,
                                            te_files, spk2idx,
                                            opts.order,
                                            opts.stats,
                                            ext=opts.ext,
                                            np_fmt=opts.np_fmt)
            te_dloader = DataLoader(te_dset, batch_size=1,
                                    shuffle=False)
            with torch.no_grad():
                teloss = []
                teacc = []
                timings = []
                beg_t = timeit.default_timer()
                if opts.test_log_file is not None:
                    test_log_f = open(opts.test_log_file, 'w')
                    test_log_f.write('Filename\tAccuracy [%]\tError [%]\n')
                else:
                    test_log_f = None
                for bidx, batch in enumerate(te_dloader, start=1):
                    #X, Y, slen = batch
                    X, Y = batch
                    X = X.transpose(1, 2).to(device)
                    Y = Y.to(device)
                    Y_ = model(X)
                    Y = Y.view(-1, 1).repeat(1, Y_.size(2))
                    loss = F.nll_loss(Y_, Y)
                    acc = accuracy(Y_, Y)
                    if test_log_f:
                        test_log_f.write('{}\t{:.2f}\t{:.2f}\n' \
                                         ''.format(te_files[bidx - 1],
                                                   acc * 100,
                                                   100 - (acc * 100)))
                    teacc.append(accuracy(Y_, Y))
                    teloss.append(loss)
                    end_t = timeit.default_timer()
                    timings.append(end_t - beg_t)
                    beg_t = timeit.default_timer()
                    if bidx % 100 == 0 or bidx == 1:
                        mteloss = np.mean(teloss)
                        mteacc = np.mean(teacc)
                        mtimings = np.mean(timings)
                    print('Processed test file {}/{} mfiletime: {:.2f} s, '
                          'macc: {:.4f}, mloss: {:.2f}'
                          ''.format(bidx, len(te_dloader), mtimings,
                                    mteacc, mteloss),
                          end='\r')
                print() 
                if test_log_f:
                    test_log_f.write('-' * 30 + '\n')
                    test_log_f.write('Test accuracy: ' \
                                     '{:.2f}\n'.format(np.mean(teacc) * 100))
                    test_log_f.write('Test error: ' \
                                     '{:.2f}\n'.format(100 - (np.mean(teacc) *100)))
                    test_log_f.write('Test loss: ' \
                                     '{:.2f}\n'.format(np.mean(teloss)))
                    test_log_f.close()
                print('Test accuracy: {:.4f}'.format(np.mean(teacc)))
                print('Test loss: {:.2f}'.format(np.mean(teloss)))


def train_epoch(dloader_, model, opt, epoch, log_freq=1, writer=None,
                device='cpu', bpe=None):
    model.train()
    if bpe is None:
        # default is just dataloader length
        bpe = len(dloader_)
    global_idx = (epoch - 1) * bpe
    timings = []
    beg_t = timeit.default_timer()
    #for bidx, batch in enumerate(dloader_, start=1):
    iterator = iter(dloader)
    for bidx in range(1, bpe + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dloader)
            batch = next(iterator)
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
        #if bidx % log_freq == 0 or bidx >= len(dloader_):
        if bidx % log_freq == 0 or bidx >= bpe:
            acc = accuracy(Y_, Y)
            log_str = 'Batch {:5d}/{:5d} (Epoch {:3d}, Gidx {:5d})' \
                      ' '.format(bidx, bpe,
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
               bpe=None, key='eval'):
    model.eval()
    with torch.no_grad():
        if bpe is None:
            # default is just dataloader length
            bpe = len(dloader_)
        eval_losses = []
        eval_accs = []
        timings = []
        beg_t = timeit.default_timer()
        #for bidx, batch in enumerate(dloader_, start=1):
        iterator = iter(dloader)
        for bidx in range(1, bpe + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dloader)
                batch = next(iterator)
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
            #if bidx % log_freq == 0 or bidx >= len(dloader_):
            if bidx % log_freq == 0 or bidx >= bpe:
                
                log_str = 'EVAL::{} Batch {:4d}/{:4d} (Epoch {:3d})' \
                          ' '.format(key, bidx, bpe,
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
    parser.add_argument('--train_guia', type=str, default=None)
    parser.add_argument('--test_guia', type=str, default=None)
    parser.add_argument('--spk2idx', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--z_bnorm', action='store_true', default=False,
                        help='Use z-norm in z, before any model (Default: '
                             'False).')
    parser.add_argument('--va_split', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--sched_mode', type=str, default='plateau',
                        help='(1) plateau (validation), (2) '
                             'step (step decay) (Def: plateau).')
    parser.add_argument('--sched_step_size', type=int,
                        default=30, help='Number of epochs to apply '
                                          'a learning rate decay (Def: 30).')
    parser.add_argument('--lrdec', type=float, default=0.1,
                        help='Decay factor of learning rate after '
                             'patience epochs of valid accuracy not '
                             'improving (Def: 0.1).')
    parser.add_argument('--test_ckpt', type=str, default=None)
    parser.add_argument('--plateau_mode', type=str, default='max',
                        help='LR Plateau scheduling mode; (1) max, (2) min '
                             '(Def: max).')
    parser.add_argument('--order', type=int, default=39)
    parser.add_argument('--stats', type=str, default=None)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--model', type=str, default='mlp',
                        help='(1) cls, (2) mlp (Def: mlp).')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ext', type=str, default='mfcc')
    parser.add_argument('--test_log_file', type=str, default=None,
                        help='Possible test log file (Def: None).')
    parser.add_argument('--np-fmt', action='store_true', default=False,
                        help='Whether or not aco files are in numpy '
                             'format (Def: False).')
    
    opts = parser.parse_args()

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    main(opts)
