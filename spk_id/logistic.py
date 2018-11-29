import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset, DataLoader
import json
import glob
from tensorboardX import SummaryWriter
import random
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
import os


class LogisticDataset(Dataset):

    def __init__(self, data_root, cfg,
                 split, rand_sample=True):
        super().__init__()
        self.data_root = data_root
        self.split = split
        with open(cfg, 'r') as cfg_f:
            self.cfg = json.load(cfg_f)
            self.spk2idx = self.cfg['spk2idx']
        self.rand_sample = rand_sample

    def __getitem__(self, index):
        npypath = self.cfg[self.split]['wav_files'][index]
        lab = self.cfg[self.split]['spk_ids'][index]
        npyfile = os.path.join(self.data_root, 
                               npypath)
        x = np.load(npyfile)
        y = self.spk2idx[lab]
        if self.rand_sample:
            x = x.T[random.choice(list(range(x.T.shape[0]))), :]
            x = x[None, :]
        return x, y

    def __len__(self):
        return len(self.cfg[self.split]['wav_files'])

def collater(batch):
    X = []
    Y = []
    for sample in batch:
        x, y = sample
        Y += [y] * x.shape[0]
        X.append(torch.tensor(x))
    return torch.cat(X, dim=0), torch.tensor(Y)

class LogisticRegression(nn.Module):

    def __init__(self, input_dim=256, num_spks=108):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_spks)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.act(self.linear(x))

def train(opts):
    CUDA = torch.cuda.is_available()
    device = 'cuda' if CUDA else 'cpu'
    dset = LogisticDataset(opts.data_root, opts.data_cfg, 'train')
    dloader = DataLoader(dset, batch_size=opts.batch_size, collate_fn=collater,
                         shuffle=True)
    model = LogisticRegression(input_dim=256, num_spks=len(dset.spk2idx))
    model.to(device)
    criterion = nn.NLLLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(opts.save_path)
    losses = []
    global_step = 0
    for e in range(opts.epoch):
        timings = []
        beg_t = timeit.default_timer()
        for bidx, batch in enumerate(dloader, start=1):
            opt.zero_grad()
            batchX, batchY = batch
            batchX = batchX.to(device)
            batchY = batchY.to(device)
            pred = model(batchX)
            loss = criterion(pred, batchY)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if bidx % opts.log_freq == 0:
                print('Batch {}/{} (Epoch {}) loss: {:.3f} mbtime: {:.3f}'
                      ' s'.format(bidx, len(dloader), e,
                                  loss.item(), np.mean(timings)))
                writer.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1
    torch.save(model.state_dict(), os.path.join(opts.save_path,
                                                'lr.ckpt'))

def test(opts):
    CUDA = torch.cuda.is_available()
    device = 'cuda' if CUDA else 'cpu'
    # for every test file, read it and compute likelihoods and accs
    with open(opts.data_cfg, 'r') as cfg_f:
        cfg = json.load(cfg_f)
        print(list(cfg.keys()))
        spk2idx = cfg['spk2idx']
        # make model
        model = LogisticRegression(input_dim=256, num_spks=len(spk2idx))
        model.to(device)
        ckpt = torch.load(os.path.join(opts.save_path, 'lr.ckpt'),
                          map_location=device)
        model.load_state_dict(ckpt)
        files = cfg['test']['wav_files']
        labs = cfg['test']['spk_ids']
        total_files = len(files)
        totuttacc = 0
        totmframeacc = 0
        accs = {}
        for fname, lab in zip(files, labs):
            spk_id = spk2idx[lab]
            data = np.load(os.path.join(opts.data_root, fname)).T
            x = torch.tensor(data).to(device)
            sy = torch.tensor([spk_id]).to(device) # single utt target
            y = torch.tensor([spk_id] * x.size(0)).to(device)
            y_ = model(x)
            # predict all labels, one per frame
            pred = y_.max(1, keepdim=True)[1]
            macc = pred.eq(y.view_as(pred)).float().mean().item()
            # predict global label
            utt_pred = torch.mean(y_, dim=0, keepdim=True).max(1, keepdim=True)[1]
            uacc = utt_pred.eq(sy.view_as(utt_pred)).float().mean().item()
            accs[fname] = {'mframe_accuracy':macc,
                           'utt_accuracy':uacc}
            totmframeacc += macc
            totuttacc += uacc
        
        totmframeacc /= total_files
        totuttacc /= total_files
        accs['total'] = {'mframe_accuracy':totmframeacc,
                         'utt_accuracy':totuttacc}

        if opts.test_log is not None:
            with open(opts.test_log, 'a') as tlog:
                tlog.write(json.dumps(accs))
        print('Model path {} mframe_acc: {:.3f}, mutt_acc:'
              ' {:.3f}'.format(opts.save_path, totmframeacc,
                               totuttacc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=None)
    parser.add_argument('--data_cfg', type=str,
                        default=None,
                        help='Dictionary containing paths to train and '
                             'test data files.')
    parser.add_argument('--save_path', type=str, default='lr_ckpt')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_log', type=str, default=None)
    parser.add_argument('--no-train', action='store_true', default=False)

    opts = parser.parse_args()
    assert opts.data_root is not None
    assert opts.data_cfg is not None
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    if not opts.no_train:
        train(opts)

    if opts.test:
        test(opts)
