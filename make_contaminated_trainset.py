from pase.dataset import *
from torchvision.transforms import Compose
import json
from pase.transforms import *
import tqdm
from torch.utils.data import DataLoader
import soundfile as sf
from train import config_distortions
import random
import numpy as np
import os
import torch

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

OUT_PATH = 'data/distorted_trainset'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
NUM_BATCHES = 10000

with open('cfg/distortions/all_x26.cfg', 'r') as dtr_cfg:
    dtr = json.load(dtr_cfg)
    #dtr['trans_p'] = opts.distortion_p
    dist_trans = config_distortions(**dtr)
    print(dist_trans)
    dataset = LibriSpeechSegTupleWavDataset
    dset = dataset('data/LibriSpeech_50h/all/', 
                   'data/librispeech_data_50h.cfg',
                   'train',
                   transform=Compose([ToTensor(),
                                      SingleChunkWav(32000,
                                                     random_scale=True)]),
                   distortion_transforms=dist_trans)
    dloader = DataLoader(dset, batch_size=100,
                         shuffle=True,
                         collate_fn=DictCollater(),
                         num_workers=4)
    iterator = iter(dloader)
    for bidx in tqdm.tqdm(range(1, NUM_BATCHES + 1), total=NUM_BATCHES):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dloader)
            batch = next(iterator)
        chunk = batch['chunk']
        for sidx in range(chunk.shape[0]):
            # scale it to properly fit in soundfile requirement
            peak = torch.abs(chunk[sidx]).max().item()
            sample = chunk[sidx].data.numpy()
            if peak > 1:
                sample = sample / peak
            sf.write(os.path.join(OUT_PATH, 'utt_{}_{}.' \
                                  'wav'.format(bidx, sidx + 1)),
                     sample[0],
                     16000)
