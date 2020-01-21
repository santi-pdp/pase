from pase.models.core import Waveminionet
from pase.models.frontend import wf_builder
from pase.dataset import PairWavDataset, DictCollater
from torchvision.transforms import Compose
from pase.transforms import *
from pase.losses import *
from pase.utils import pase_parser
from torch.utils.data import DataLoader
import tqdm
import torch
import pickle
import torch.nn as nn
import soundfile as sf
import numpy as np
import argparse
import os
import sys
import json
import random


def remove_Dcfg(minions_cfg):
    # remove unnecessary Discriminator config if found
    # in any of the minion fields
    for m_i, mcfg in enumerate(minions_cfg):
        if 'DNet_cfg' in mcfg:
            print('Removing DNet_cfg')
            del mcfg['DNet_cfg']
        if 'Dopt_cfg' in mcfg:
            print('Removing Dopt_cfg')
            del mcfg['Dopt_cfg']

def main(opts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # build network
    minions_cfg = pase_parser(opts.minions_cfg, do_losses=False)
    remove_Dcfg(minions_cfg)
    pase = wf_builder(opts.cfg)
    model = Waveminionet(minions_cfg=minions_cfg,
                         num_devices=0,
                         pretrained_ckpts=opts.ckpt,
                         z_minion=False,
                         frontend=pase)
    model.eval()
    model.to(device)
    transf = Reverb(['data/omologo_revs/IRs_2/IR_223108.imp'], ir_fmt='imp')
    minion = model.minions[0]
    minion.loss = None
    pase = model.frontend
    #print(opts.in_files)
    in_files = [os.path.join(opts.files_root, inf) for inf in opts.in_files]
    wavs = []
    wfiles =  []
    max_len = 0
    print('Total batches: ', len(in_files) // opts.batch_size)
    with torch.no_grad():
        for wi, wfile in tqdm.tqdm(enumerate(in_files, start=1),
                                   total=len(in_files)):
            wfiles.append(wfile)
            wav, rate = sf.read(wfile)
            wavs.append(wav)
            if len(wav) > max_len:
                max_len = len(wav)
            if wi % opts.batch_size == 0 or wi >= len(in_files):
                lens = []
                batch = []
                for bi in range(len(wavs)):
                    P_ = max_len - len(wavs[bi])
                    lens.append(len(wavs[bi]))
                    if P_ > 0:
                        pad = np.zeros((P_))
                        wav_ = np.concatenate((wavs[bi], pad), axis=0)
                    else:
                        wav_ = wavs[bi]
                    wav = torch.FloatTensor(wav_)
                    wav_r = transf({'chunk':wav})
                    batch.append(wav_r['chunk'].view(1, 1, -1))
                batch = torch.cat(batch, dim=0)
                x = batch.to(device)
                h = pase(x)
                #print('frontend size: ', h.size())
                y = minion(h).cpu()
                for bi in range(len(wavs)):
                    bname = os.path.basename(wfiles[bi])
                    y_ = y[bi].squeeze().data.numpy()
                    y_ = y_[:lens[bi]]
                    sf.write(os.path.join(opts.out_path, 
                                          '{}'.format(bname)),
                             y_, 16000)
                    x_ = x[bi].squeeze().data.numpy()
                    x_ = x_[:lens[bi]]
                    sf.write(os.path.join(opts.out_path, 
                                          'input_{}'.format(bname)),
                             x_, 16000)
                max_len = 0
                wavs = []
                wfiles =  []
                batch = None

    """
    with open('data/librispeech_stats.pkl', 'rb') as stats_f:
        stats = pickle.load(stats_f)
        st = stats['chunk']
        y_ = y * st['std'] + st['mean']
        y_ = y_.squeeze().data.numpy()
        sf.write('reconchunk_{}'.format(bname), y_, 16000)
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minions_cfg', type=str,
                        default='cfg/all_RF6250_for_distorted_adversarial.cfg')
    parser.add_argument('--cfg', type=str, default='cfg/PASE_RF6250.cfg')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--out_path', type=str, default='chunk_generations')
    parser.add_argument('--in_files', type=str, default=None, nargs='+')
    parser.add_argument('--files_root', type=str, default='.')
    parser.add_argument('--stats', type=str,
                        default='data/librispeech_stats_nochunks.pkl')

    opts = parser.parse_args()
    if not os.path.exists(opts.out_path):
        os.makedirs(opts.out_path)
    main(opts)
