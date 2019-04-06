from pase.models.core import Waveminionet
from pase.dataset import PairWavDataset, DictCollater
from torchvision.transforms import Compose
from pase.transforms import *
from pase.losses import *
from pase.utils import pase_parser
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch
import pickle
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import random


def eval(opts):
    CUDA = True if torch.cuda.is_available() and not opts.no_cuda else False
    device = 'cuda' if CUDA else 'cpu'
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if CUDA:
        torch.cuda.manual_seed_all(opts.seed)
    print('Seeds initialized to {}'.format(opts.seed))
    # ---------------------
    # Transforms
    trans = Compose([
        ToTensor(),
        MIChunkWav(opts.chunk_size, random_scale=opts.random_scale),
        LPS(opts.nfft, hop=opts.stride, win=400),
        MFCC(hop=opts.stride),
        Prosody(hop=opts.stride, win=400),
        ZNorm(opts.stats)
    ])
    print(trans)

    # ---------------------
    # Build Dataset(s) and DataLoader(s)
    dset = PairWavDataset(opts.data_root, opts.data_cfg, 'valid',
                         transform=trans)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=False, collate_fn=DictCollater(),
                         num_workers=opts.num_workers)
    # Compute estimation of bpe. As we sample chunks randomly, we
    # should say that an epoch happened after seeing at least as many
    # chunks as total_train_wav_dur // chunk_size
    bpe = (dset.total_wav_dur // opts.chunk_size) // opts.batch_size

    # ---------------------
    # Build Model
    if opts.fe_cfg is not None:
        with open(opts.fe_cfg, 'r') as fe_cfg_f:
            fe_cfg = json.load(fe_cfg_f)
            print(fe_cfg)
    else:
        fe_cfg = None
    model = Waveminionet(minions_cfg=pase_parser(opts.net_cfg),
                         adv_loss=opts.adv_loss,
                         pretrained_ckpt=opts.pretrained_ckpt,
                         frontend_cfg=fe_cfg
                        )

    print(model)
    model.to(device)
    writer = SummaryWriter(opts.save_path)
    if opts.max_epoch is not None:
        # just make a sequential search til max epoch ckpts
        ckpts = ['fullmodel_e{}.ckpt'.format(e) for e in range(opts.max_epoch)]
    else:
        ckpts = opts.ckpts
    for model_ckpt in ckpts:
        # name format is fullmodel_e{}.ckpt
        epoch = int(model_ckpt.split('_')[-1].split('.')[0][1:])
        model_ckpt = os.path.join(opts.ckpt_root,
                                  model_ckpt)
        print('Loading ckpt ', model_ckpt)
        model.load_pretrained(model_ckpt, load_last=True, verbose=False)
        model.eval_(dloader, opts.batch_size, 
                    bpe, log_freq=opts.log_freq,
                    epoch_idx=epoch, writer=writer,
                    device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='data/VCTK')
    parser.add_argument('--data_cfg', type=str, 
                        default='data/vctk_data.cfg')
    parser.add_argument('--net_cfg', type=str,
                        default=None)
    parser.add_argument('--fe_cfg', type=str, default=None)
    parser.add_argument('--ckpt_root', type=str, default='.')
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--ckpts', type=str, nargs='+', default=None)
    parser.add_argument('--stats', type=str, default='data/vctk_stats.pkl')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='ckpt')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--random_scale', action='store_true', default=False)
    parser.add_argument('--chunk_size', type=int, default=16000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--nfft', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--stride', type=int, default=160)
    parser.add_argument('--fe_opt', type=str, default='Adam')
    parser.add_argument('--min_opt', type=str, default='Adam')
    parser.add_argument('--dout', type=float, default=0.2)
    parser.add_argument('--fe_lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=0.0004)
    parser.add_argument('--z_lr', type=float, default=0.0004)
    parser.add_argument('--rndmin_train', action='store_true',
                        default=False)
    parser.add_argument('--adv_loss', type=str, default='BCE',
                        help='BCE or L2')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Epoch to begin applying z adv '
                             '(Def: 2).')
    parser.add_argument('--zinit_weight', type=float, default=1)
    parser.add_argument('--zinc', type=float, default=0.0002)

    opts = parser.parse_args()
    if opts.net_cfg is None:
        raise ValueError('Please specify a net_cfg file')

    eval(opts)
