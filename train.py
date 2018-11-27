from waveminionet.models.core import Waveminionet
from waveminionet.dataset import PairWavDataset, DictCollater
from torchvision.transforms import Compose
from waveminionet.transforms import *
from waveminionet.losses import *
from torch.utils.data import DataLoader
import torch
import pickle
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import random


def train(opts):
    CUDA = True if torch.cuda.is_available() and not opts.no_cuda else False
    device = 'cuda' if CUDA else 'cpu'
    num_devices = 1
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if CUDA:
        torch.cuda.manual_seed_all(opts.seed)
        num_devices = torch.cuda.device_count()
        print('[*] Using CUDA {} devices'.format(num_devices))
    else:
        print('[!] Using CPU')
    print('Seeds initialized to {}'.format(opts.seed))

    model = Waveminionet(minions_cfg=[
                          {'num_outputs':1,
                           'dropout':0.2,
                           'name':'chunk',
                           'type':'decoder',
                           'loss':nn.MSELoss()
                          },
                          {'num_outputs':1025,
                           'dropout':0.2,
                           'name':'lps',
                           'loss':nn.MSELoss()
                          },
                          {'num_outputs':20,
                           'dropout':0.2,
                           'name':'mfcc',
                           'loss':nn.MSELoss()
                          },
                          {'num_outputs':4,
                           'dropout':0.2,
                           'name':'prosody',
                           'loss':nn.MSELoss()
                          },
                          {'num_outputs':1,
                           'dropout':0.2,
                           'name':'mi',
                           'loss':nn.BCEWithLogitsLoss(),
                           'skip':False,
                           'keys':['chunk',
                                   'chunk_ctxt',
                                   'chunk_rand']
                          }], 
        adv_loss=opts.adv_loss,
        num_devices=num_devices)
    if opts.pretrained_ckpt is not None:
        model.load_pretrained(opts.pretrained_ckpt, load_last=True)
    print(model)
    model.to(device)
    trans = Compose([
        ToTensor(),
        MIChunkWav(opts.chunk_size),
        LPS(),
        MFCC(),
        Prosody(),
        ZNorm(opts.stats)
    ])
    print(trans)
    dset = PairWavDataset(opts.data_root, opts.data_cfg, 'train',
                         transform=trans)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, collate_fn=DictCollater(),
                         num_workers=opts.num_workers)
    # Compute estimation of bpe. As we sample chunks randomly, we
    # should say that an epoch happened after seeing at least as many
    # chunks as total_train_wav_dur // chunk_size
    bpe = (dset.total_wav_dur // opts.chunk_size) // opts.batch_size
    opts.bpe = bpe
    # fastet lr to MI
    #opts.min_lrs = {'mi':0.001}
    model.train(dloader, vars(opts), device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='data/VCTK')
    parser.add_argument('--data_cfg', type=str, 
                        default='data/vctk_data.cfg')
    parser.add_argument('--stats', type=str, default='data/vctk_stats.pkl')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='ckpt')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--chunk_size', type=int, default=16000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--fe_opt', type=str, default='Adam')
    parser.add_argument('--min_opt', type=str, default='Adam')
    parser.add_argument('--fe_lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=0.0004)
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
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))
    train(opts)
