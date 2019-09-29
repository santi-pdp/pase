import json
import librosa
import argparse
import random
from random import shuffle
import numpy as np
import os
import tqdm
from functools import partial, reduce
from multiprocessing import Pool


def get_file_dur(fname):
    x, rate = librosa.load(fname, sr=None)
    return len(x)

def get_file_dur_join(fname, data_root):
    return get_file_dur(os.path.join(data_root, fname))


def main(opts):
    random.seed(opts.seed)
    spk2idx = np.load(opts.libri_dict, allow_pickle=True)
    spk2idx = dict(spk2idx.any())
    data_cfg = {'train':{'data':[],
                         'speakers':[]},
                'valid':{'data':[],
                         'speakers':[]},
                'test':{'data':[],
                        'speakers':[]},
                'speakers':[]}
    with open(opts.train_scp, 'r') as train_f:
        train_files = [l.rstrip() for l in train_f]
        shuffle(train_files)
        N_valid_files = int(len(train_files) * opts.val_ratio)
        valid_files = train_files[:N_valid_files]
        train_files = train_files[N_valid_files:]
        for ti, train_file in enumerate(train_files, start=1):
            spk = spk2idx[train_file]
            if spk not in data_cfg['speakers']:
                data_cfg['speakers'].append(spk)
                data_cfg['train']['speakers'].append(spk)
            data_cfg['train']['data'].append({'filename':train_file,
                                              'spk':spk})
        with Pool(opts.num_workers) as p:
            train_dur = list(tqdm.tqdm(p.imap(partial(get_file_dur_join, data_root=opts.data_root), train_files), total=len(train_files)))
        train_dur = reduce(lambda x, y: x + y, train_dur, 0)
        data_cfg['train']['total_wav_dur'] = train_dur
        print()

        valid_dur = 0
        for ti, valid_file in enumerate(valid_files, start=1):
            print('Processing valid file {:7d}/{:7d}'.format(ti,
                                                             len(valid_files)),
                  end='\r')
            spk = spk2idx[valid_file]
            if spk not in data_cfg['speakers']:
                data_cfg['speakers'].append(spk)
                data_cfg['valid']['speakers'].append(spk)
            data_cfg['valid']['data'].append({'filename':valid_file,
                                              'spk':spk})
        with Pool(opts.num_workers) as p:
            valid_dur = list(tqdm.tqdm(p.imap(partial(get_file_dur_join, data_root=opts.data_root), valid_files), total=len(valid_files)))            
        valid_dur = reduce(lambda x, y: x + y, valid_dur, 0)
        data_cfg['valid']['total_wav_dur'] = valid_dur
        print()

    with open(opts.test_scp, 'r') as test_f:
        test_files = [l.rstrip() for l in test_f]
        test_dur = 0
        for ti, test_file in enumerate(test_files, start=1):
            print('Processing test file {:7d}/{:7d}'.format(ti,
                                                            len(test_files)),
                  end='\r')
            spk = spk2idx[test_file]
            if spk not in data_cfg['speakers']:
                data_cfg['speakers'].append(spk)
                data_cfg['test']['speakers'].append(spk)
            data_cfg['test']['data'].append({'filename':test_file,
                                              'spk':spk})

        with Pool(opts.num_workers) as p:
            test_dur = list(tqdm.tqdm(p.imap(partial(get_file_dur_join, data_root=opts.data_root), test_files), total=len(test_files)))            
        test_dur = reduce(lambda x, y: x + y, test_dur, 0)
        data_cfg['test']['total_wav_dur'] = test_dur
    print()

    with open(opts.cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))


            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='data/LibriSpeech/Librispeech_spkid_sel')
    parser.add_argument('--train_scp', type=str, default=None)
    parser.add_argument('--test_scp', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio to take out of training '
                             'in utterances ratio (Def: 0.1).')
    parser.add_argument('--cfg_file', type=str, default='data/librispeech_data.cfg')
    parser.add_argument('--libri_dict', type=str, default='data/LibriSpeech/libri_dict.npy')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    
    opts = parser.parse_args()
    main(opts)

