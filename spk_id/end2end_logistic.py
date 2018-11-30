import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from waveminionet.models.frontend import WaveFe
from waveminionet.dataset import WavDataset, DictCollater
from torchvision.transforms import Compose
from waveminionet.transforms import *
from waveminionet.losses import *
from torch.utils.data import DataLoader
import json
import glob
from tensorboardX import SummaryWriter
import random
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
import os

class E2ELogisticRegression(nn.Module):

    def __init__(self, input_dim=256, num_spks=108):
        super().__init__()
        self.frontend = WaveFe()
        ninp = self.frontend.emb_dim
        self.linear = nn.Conv1d(ninp, num_spks, 1)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.act(self.linear(self.frontend(x)))

def train(opts):
    CUDA = torch.cuda.is_available()
    device = 'cuda' if CUDA else 'cpu'
    trans = Compose([
        ToTensor(),
        SingleChunkWav(opts.chunk_size)
    ])
    print(trans)
    dset = WavDataset(opts.data_root, opts.data_cfg, 'train',
                      transform=trans, return_spk=True)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, collate_fn=DictCollater(),
                         num_workers=opts.num_workers)
    va_dset = WavDataset(opts.data_root, opts.data_cfg, 'valid',
                         transform=trans, return_spk=True)
    va_dloader = DataLoader(dset, batch_size=opts.batch_size,
                            shuffle=False, collate_fn=DictCollater(),
                            num_workers=opts.num_workers)
    # Compute estimation of bpe. As we sample chunks randomly, we
    # should say that an epoch happened after seeing at least as many
    # chunks as total_train_wav_dur // chunk_size
    bpe = (dset.total_wav_dur // opts.chunk_size) // opts.batch_size
    print('Batches per epoch: ', bpe)
    model = E2ELogisticRegression(input_dim=256, num_spks=len(dset.spk2idx))
    print(model)
    model.to(device)
    criterion = nn.NLLLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(opts.save_path)
    losses = []
    global_step = 0
    min_val_loss = np.inf
    for e in range(opts.epoch):
        model.train()
        timings = []
        beg_t = timeit.default_timer()
        for bidx in range(1, bpe + 1):
            batch = next(dloader.__iter__())
            opt.zero_grad()
            batchX, batchY = batch
            batchX = batchX['chunk']
            batchX = batchX.to(device)
            pred = model(batchX)
            # replicate label to match the output time shape
            batchY = batchY.unsqueeze(1).repeat(1, pred.size(2))
            batchY = batchY.to(device)
            loss = criterion(pred, batchY)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if bidx % opts.log_freq == 0:
                # predict all labels, one per frame
                y_ = pred
                y = batchY
                pred = y_.max(1, keepdim=True)[1]
                macc = pred.eq(y.view_as(pred)).float().mean().item()
                # predict global label
                utt_pred = torch.mean(y_, dim=2, keepdim=True).max(1, keepdim=True)[1]
                uacc = utt_pred.eq(y[:, 0].view_as(utt_pred)).float().mean().item()
                print('Batch {}/{} (Epoch {}) loss: {:.3f}, acc: {:.3f},'
                      'mbtime: {:.3f} s'
                      ''.format(bidx, bpe, e,
                                loss.item(), uacc, np.mean(timings)))
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/macc', macc, global_step)
                writer.add_scalar('train/uacc', uacc, global_step)
            global_step += 1
        val_loss = eval(va_dloader, model, criterion, e, device, writer)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(opts.save_path,
                                                        'bestval-{:.3f}_'
                                                        'end2end_lr_e{}.ckpt'
                                                        ''.format(val_loss, e)))
        torch.save(model.state_dict(), os.path.join(opts.save_path,
                                                    'end2end_lr_e{}.ckpt'
                                                    ''.format(e)))
    torch.save(model.state_dict(), os.path.join(opts.save_path,
                                                'end2end_lr.ckpt'))

def eval(dloader, model, criterion, epoch, device, writer):
    model.eval()
    losses = []
    macc = []
    uacc = []
    timings = []
    beg_t = timeit.default_timer()
    for bidx, batch in enumerate(dloader, start=1):
        batchX, batchY = batch
        batchX = batchX['chunk']
        batchX = batchX.to(device)
        pred = model(batchX)
        # replicate label to match the output time shape
        batchY = batchY.unsqueeze(1).repeat(1, pred.size(2))
        batchY = batchY.to(device)
        loss = criterion(pred, batchY)
        losses.append(loss.item())
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        y_ = pred
        y = batchY
        pred = y_.max(1, keepdim=True)[1]
        macc.append(pred.eq(y.view_as(pred)).float().mean().item())
        # predict global label
        utt_pred = torch.mean(y_, dim=2, keepdim=True).max(1, keepdim=True)[1]
        uacc.append(utt_pred.eq(y[:, 0].view_as(utt_pred)).float().mean().item())
        print('Eval batch {}/{}, mbtime: {:.3f} s'.format(bidx,
                                                          len(dloader),
                                                          np.mean(timings)))
    print('Epoch {} eval >> loss: {:.3f}, acc: {:.3f},'
          'mbtime: {:.3f} s'
          ''.format(epoch,
                    np.mean(losses), np.mean(uacc), 
                    np.mean(timings)))
    writer.add_scalar('eval/loss', np.mean(losses), epoch)
    writer.add_scalar('eval/macc', np.mean(macc), epoch)
    writer.add_scalar('eval/uacc', np.mean(uacc), epoch)
    return np.mean(losses)



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
    parser.add_argument('--save_path', type=str, default='ckpt_end2endlr')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_log', type=str, default=None)
    parser.add_argument('--no-train', action='store_true', default=False)
    parser.add_argument('--chunk_size', type=int, default=16000)
    parser.add_argument('--num_workers', type=int, default=0)

    opts = parser.parse_args()
    assert opts.data_root is not None
    assert opts.data_cfg is not None
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    if not opts.no_train:
        train(opts)

    if opts.test:
        test(opts)
