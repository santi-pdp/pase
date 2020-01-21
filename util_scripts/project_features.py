import numpy as np
from tensorboardX import SummaryWriter
import json
import random
random.seed(1)
from random import shuffle
import torch
import tqdm
import os
import glob


SAVE_PATH= 'vctk_projection_paseQRNN_age'
#SAVE_PATH= 'vctk_projection_paseQRNN_id'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
FEATS_DIR='vc/data/vctk/all_trimmed_paseQRNNpase-eval_vctk_raw/mel/'
spk_info = json.load(open('vc/data/vctk/speaker-info.json', 'r'))
print('Found {} speakers'.format(len(spk_info)))
files = glob.glob(os.path.join(FEATS_DIR, '*.npy'))
print('Found {} files'.format(len(files)))
MAX_SPK_SAMPLES = 10000000
#MAX_SPK_SAMPLES = 50
#MAX_SPKS = 30
MAX_SPKS = None
#LABEL = 'ACCENTS'
LABEL = 'AGE'
#LABEL = 'GENDER'
#LABEL = None
shuffle(files)

spk2count = {}
spk2idx = {}
labels = []
feats = []

for fi, filename in tqdm.tqdm(enumerate(files, start=1), total=len(files)):
    bname = os.path.basename(filename)
    spkname = bname.split('_')[0]
    if MAX_SPKS is not None and len(spk2count) >= MAX_SPKS:
        if spkname not in spk2idx:
            continue
    if spkname not in spk2idx:
        spk2count[spkname] = 0
        spk2idx[spkname] = len(spk2idx)
    data = np.load(filename)
    avg = np.mean(data, axis=1)
    spk2count[spkname] += 1
    if spk2count[spkname] < MAX_SPK_SAMPLES:
        feats.append(avg[None, :])
        try:
            lab = spk_info[spkname][LABEL]
        except KeyError:
            lab = spk2idx[spkname]
        labels.append([lab])
    if fi >= 1807:
        break

feats = torch.FloatTensor(np.concatenate(feats, axis=0))
labels = np.concatenate(labels, axis=0)
print('feats shape: ', feats.shape)
print('labels shape: ', labels.shape)
writer = SummaryWriter(SAVE_PATH)
writer.add_embedding(feats, metadata=labels)



