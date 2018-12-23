import torch
from torch.utils.data import DataLoader
from waveminionet.models.frontend import WaveFe
from waveminionet.dataset import WavDataset, uttwav_collater
import pickle
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import random
import timeit


def extract(opts):
    CUDA = True if torch.cuda.is_available() and not opts.no_cuda else False
    device = 'cuda' if CUDA else 'cpu'
    num_devices = 1
    if CUDA:
        num_devices = torch.cuda.device_count()
        print('[*] Using CUDA {} devices'.format(num_devices))
    else:
        print('[!] Using CPU')
    # Build the frontend
    model = WaveFe()
    model.load_pretrained(opts.ckpt, load_last=True)
    model.to(device)
    model.eval()

    # Build dataset and dataloader
    dset = WavDataset(opts.data_root, opts.data_cfg, opts.split,
                      return_uttname=True)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=False, 
                         num_workers=opts.num_workers,
                         collate_fn=uttwav_collater)

    timings = []
    beg_t = timeit.default_timer()
    # Go sample by sample
    for widx, sample in enumerate(dloader, start=1):
        wavs, uttnames, lens = sample
        wavs = wavs.to(device)
        z = model(wavs.unsqueeze(1))
        for s_i in range(wavs.size(0)):
            if opts.merge_uttname:
                uname = uttnames[s_i].replace('/', '_')
                uttname = os.path.splitext(uname)[0]
            else:
                uttname = os.path.splitext(os.path.basename(uttnames[s_i]))[0]
            out_name = os.path.join(opts.save_path,
                                    opts.split,
                                    uttname)
            maxlen = int(np.ceil(lens[s_i] / 80))
            z_sample = z[s_i].data.cpu().numpy()
            z_sample = z_sample[:, :maxlen]
            np.save(out_name, z_sample) 
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        if widx % opts.log_freq == 0 or widx >= len(dloader):
            print('Extracted wav {:5d}/{:5d} z_size: {}, mtime: {:.2f} s'
                  ''.format(widx, len(dloader), z.size(),
                            np.mean(timings)))
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='data/VCTK')
    parser.add_argument('--data_cfg', type=str, 
                        default='data/vctk_data.cfg')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='fefeats')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--merge_uttname', action='store_true', default=False,
                        help='Dont take basename, but merge with underscores '
                             'uttnames (TFCommands format).')


    opts = parser.parse_args()
    if not os.path.exists(os.path.join(opts.save_path, opts.split)):
        os.makedirs(os.path.join(opts.save_path, opts.split))

    assert opts.ckpt is not None
    extract(opts)
