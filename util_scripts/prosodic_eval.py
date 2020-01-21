from pase.models.core import Waveminionet
from pase.dataset import PairWavDataset, DictCollater
from torchvision.transforms import Compose
from pase.transforms import *
from pase.losses import *
from pase.utils import pase_parser
from torch.utils.data import DataLoader
import torch
import pickle
import timeit
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import random
from ahoproc_tools.error_metrics import RMSE, AFPR


def z_denorm(x, stats, device):
    mean = torch.FloatTensor(stats['mean']).to(device)
    std = torch.FloatTensor(stats['std']).to(device)
    return x * std + mean

def forward_dloader(dloader, bpe, fe, pmodel, stats, tags, device):
    proso_res = dict((k, []) for k in tags)
    timings = []
    beg_t = timeit.default_timer()
    for bidx in range(bpe):
        batch = next(dloader.__iter__())
        with torch.no_grad():
            Y = batch['prosody'].to(device)
            fe_h = fe(batch['chunk'].to(device))
            Y_ = pmodel(fe_h)
        nfeats = Y.size(1)
        Y = Y.transpose(1, 2).contiguous().view(-1, nfeats)
        Y_ = Y_.transpose(1, 2).contiguous().view(-1, nfeats)
        Y_ = z_denorm(Y_, stats['prosody'], device)
        loss = pmodel.loss(Y_, Y)
        # F0
        pf0 = torch.exp(Y_[:, 0]).data.cpu().numpy()
        gf0 = torch.exp(Y[:, 0]).data.cpu().numpy()
        f0_rmse = RMSE(pf0, gf0)
        proso_res['lf0'] = np.asscalar(f0_rmse)
        # ZCR
        pzcr = Y_[:, 2].data.cpu().numpy()
        gzcr = Y[:, 2].data.cpu().numpy()
        zcr_rmse = RMSE(pzcr, gzcr)
        proso_res['zcr'] = np.asscalar(zcr_rmse)
        # ENERGY
        pegy = Y_[:, 3].data.cpu().numpy()
        gegy = Y[:, 3].data.cpu().numpy()
        egy_rmse = RMSE(pegy, gegy)
        proso_res['egy'] = np.asscalar(egy_rmse)
        # U/V FLAG
        puv = np.round(Y_[:, 1].data.cpu().numpy())
        guv = Y[:, 1].data.cpu().numpy()
        uv_afpr = AFPR(puv, guv)[0]
        proso_res['uv'] = np.asscalar(uv_afpr)
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        if (bidx + 1) % 10 == 0 or (bidx + 1) >= len(dloader):
            print('Done batches {:5d}/{:5d}, mtime: {:.2f} s'
                  ''.format(bidx + 1, len(dloader), np.mean(timings)),
                 end='\r')
    print('')

    return proso_res

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
        Prosody(hop=opts.stride, win=400)
    ])

    with open(opts.stats, 'rb') as stats_f:
        stats = pickle.load(stats_f)

    # ---------------------
    # Build Dataset(s) and DataLoader(s)
    dset = PairWavDataset(opts.data_root, opts.data_cfg, 'test',
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
                         frontend_cfg=fe_cfg
                        )

    print(model)
    model.to(device)

    ckpts = opts.ckpts
    use_epid = False
    if opts.ckpt_epochs is not None:
        use_epid = True
        ckpts = opts.ckpt_epochs
    if ckpts is None:
        raise ValueError('Please specify either ckpts or ckpt_epochs')

    if opts.ckpt_root is None:
        raise ValueError('Please specify ckpt_root!')

    ckpts_res = []

    for ckpt in ckpts:
        if use_epid:
            ckpt_name = 'fullmodel_e{}.ckpt'.format(ckpt)
        else:
            ckpt_name = ckpt
        ckpt_path = os.path.join(opts.ckpt_root, ckpt_name)
        print('Loading ckpt: ', ckpt_path)
        model.load_pretrained(ckpt_path, load_last=True, verbose=True)

        # select prosodic minion
        pmodel = None
        for minion in model.minions:
            if 'prosody' in minion.name:
                pmodel = minion

        # select frontend
        fe = model.frontend

        ckpts_res.append(forward_dloader(dloader, bpe, fe, pmodel, 
                                         stats, opts.tags, device))
        print('Results for ckpt {}'.format(ckpt_name))
        print('-' * 25)
        for k, v in ckpts_res[-1].items():
            print('{}: {}'.format(k, np.mean(v)))
        print('=' * 25)

    with open(opts.out_file, 'w') as out_f:
        out_f.write(json.dumps(ckpts_res, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str, 
                        default='res_prosodic_eval.json')
    parser.add_argument('--data_root', type=str, 
                        default='data/VCTK')
    parser.add_argument('--data_cfg', type=str, 
                        default='data/vctk_data.cfg')
    parser.add_argument('--net_cfg', type=str,
                        default=None)
    parser.add_argument('--fe_cfg', type=str, default=None)
    parser.add_argument('--ckpt_root', type=str, default='.')
    parser.add_argument('--ckpts', type=str, nargs='+', 
                        default=None)
    parser.add_argument('--ckpt_epochs', type=str, nargs='+', 
                        default=None)
    parser.add_argument('--stats', type=str, default='data/vctk_stats.pkl')
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
    parser.add_argument('--stride', type=int, default=160)
    parser.add_argument('--tags', type=str, nargs='+',
                        default=['lf0', 'uv', 'zcr', 'egy'])

    opts = parser.parse_args()
    if opts.net_cfg is None:
        raise ValueError('Please specify a net_cfg file')

    eval(opts)
