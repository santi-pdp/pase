import json
import librosa
import argparse
import random
from random import shuffle
import numpy as np
import os

def get_file_dur(fname):
    x, rate = librosa.load(fname, sr=None)
    return len(x)

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
        train_dur = 0
        for ti, train_file in enumerate(train_files, start=1):
            print('Processing train file {:7d}/{:7d}'.format(ti,
                                                             len(train_files)),
                  end='\r')
            spk = spk2idx[train_file]
            if spk not in data_cfg['speakers']:
                data_cfg['speakers'].append(spk)
                data_cfg['train']['speakers'].append(spk)
            data_cfg['train']['data'].append({'filename':train_file,
                                              'spk':spk})
            train_dur += get_file_dur(os.path.join(opts.data_root,
                                                   train_file))
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
            valid_dur += get_file_dur(os.path.join(opts.data_root,
                                                   valid_file))
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
            test_dur += get_file_dur(os.path.join(opts.data_root,
                                                  test_file))
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
    parser.add_argument('--libri_dict', type=str,
                        default='data/LibriSpeech/libri_dict.npy')
    parser.add_argument('--seed', type=int, default=3)
    
    opts = parser.parse_args()
    main(opts)

