import json
import numpy as np
import random
random.seed(3)
from random import shuffle
from collections import defaultdict
import os


# filenames
# ########################
# format is  llSxXyyy.lsf
# LL : language  (fr)
# S : style 
# x : session number (1/2)
# X : M or F (male or female)
# yyy : item number
# l : Linear
# sf : sampling frequency

INSTANCE="data/interface/inter1en/all_wav/"
OUT_DICT="data/interface/inter1en/interface_dict.npy"
TRAIN_SCP="data/interface/inter1en/train.scp"
TEST_SCP="data/interface/inter1en/test.scp"
OUT_SENT2IDX="data/interface/inter1en/sent2idx.json"

wavs = os.listdir(INSTANCE)

def get_label(filename):
    # just obtain style flag
    style = filename[2]
    return style

sent2idx = {}
iface_dict = {}
sent2files = defaultdict(list)

for wav in wavs:
    bname = os.path.basename(wav)
    label = get_label(bname)
    if label not in sent2idx:
        sent2idx[label] = len(sent2idx)
    bname = os.path.splitext(bname)[0]
    iface_dict[wav] = sent2idx[label]
    sent2files[label].append(wav)

print(json.dumps(sent2idx, indent=2))

with open(OUT_SENT2IDX, 'w') as sent2idx_f:
    sent2idx_f.write(json.dumps(sent2idx, indent=2))

np.save(OUT_DICT, iface_dict)

# make splits per sentiment
TRAIN_SPLIT=0.9
train_scp = []
test_scp = []

for sent, files in sent2files.items():
    shuffle(files)
    train_N = int(TRAIN_SPLIT * len(files))
    train_files = files[:train_N]
    test_files = files[train_N:]
    train_scp += train_files
    test_scp += test_files

print('Len train files: ', len(train_scp))
print('Len test files: ', len(test_scp))

with open(TRAIN_SCP, 'w') as train_f:
    for tr_fname in train_scp:
        train_f.write(tr_fname + '\n')

with open(TEST_SCP, 'w') as test_f:
    for te_fname in test_scp:
        test_f.write(te_fname + '\n')
