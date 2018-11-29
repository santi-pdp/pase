import json
import glob
import os


DATA_PATH = '../fefeats/bsz16/epoch0'
epoch = list(range(0, 13))
splits = ['train', 'test', 'valid']
MAX_WAVS_SPK = {'train':100,
                'test':10,
                'valid':10}
spk2count = {}
cfg = {}

splits = ['train', 'test', 'valid']
spk2split = {}#0
spk2idx = {}
dataset = glob.glob('{}/all/*.npy'.format(DATA_PATH))
for filename in dataset:
    fname = os.path.basename(filename)
    bname = os.path.splitext(fname)[0]
    spk_id = bname.split('_')[0]
    if spk_id not in spk2count:
        spk2count[spk_id] = {'train':0,
                             'test':0,
                             'valid':0}
        spk2split[spk_id] = 0
        spk2idx[spk_id] = len(spk2idx)
    curr_split = spk2split[spk_id]
    curr_samples = spk2count[spk_id][splits[curr_split]]
    if  curr_samples >= MAX_WAVS_SPK[splits[curr_split]]:
        if curr_split >= len(splits) - 1:
            continue
        spk2split[spk_id] += 1
    else:
        if splits[curr_split] not in cfg:
            cfg[splits[curr_split]] = {'wav_files':[],
                                       'spk_ids':[]}
        cfg[splits[curr_split]]['wav_files'].append(fname)
        cfg[splits[curr_split]]['spk_ids'].append(spk_id)
        spk2count[spk_id][splits[curr_split]] += 1
cfg['spk2idx'] = spk2idx

with open('bsz16_fefeats_data.cfg', 'w') as cfg_f:
    cfg_f.write(json.dumps(cfg, indent=2))

