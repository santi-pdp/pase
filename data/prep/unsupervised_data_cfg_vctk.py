import json
import glob
import os
import argparse
import re
import numpy as np
import librosa
import timeit


def main(opts):
    data_root = opts.data_root
    if data_root is None:
        raise ValueError('Please specify a data_root directory where '
                         'VCTK is located, containing its wav/wav16 '
                         'sub-dir and speaker-info.txt')
    cfg_file = opts.cfg_file
    prog = re.compile('\s+')
    spk_info_fname = os.path.join(data_root, 'speaker-info.txt')
    header = []
    idx2head = {}
    # store spks info dictionary
    spks = {}
    with open(spk_info_fname, 'r') as spk_info_f:
        for li, line in enumerate(spk_info_f, start=1):
            content = prog.split(line.rstrip())
            if li == 1:
                header = [h for h in content]
            else:
                if len(content) > len(header):
                    # merge last elements for they are
                    # many-word regions
                    content = content[:len(header) - 1] + \
                        ['_'.join(content[len(header) - 1:])]
                elif len(content) < len(header):
                    content += ['UNK']
                assert len(content) == len(header), print(content)
                spks[content[0]] = dict((k, v) for k, v in zip(header[1:],
                                                               content[1:]))
    # We have the speakers cfg section, now let's build the data split by spks
    spk_ids = list(spks.keys())
    N = len(spk_ids)
    train_N = int(np.floor(opts.train_split * N))
    valid_N = int(np.floor(opts.valid_split * N))
    test_split = 1. - (opts.train_split + opts.valid_split)
    test_N = int(np.ceil(test_split * N))
    print('Speakers splits')
    print('-' * 30)
    print('train_N: {}, valid_N: {}, test_N: {}'.format(train_N, valid_N,
                                                        test_N))
    data_cfg = {'train':{'data':[],
                         'speakers':[]},
                'valid':{'data':[],
                         'speakers':[]},
                'test':{'data':[],
                        'speakers':[]},
                'speakers':spks}

    if os.path.exists(os.path.join(data_root, 'wav16')):
        WAV_DIR = 'wav16'
    else:
        # By default point to 48KHz wavs
        print('WARNING: Using 48KHz wavs as no \'wav16\' dir was found!')
        WAV_DIR = 'wav48'

    splits = ['train', 'valid', 'test']
    splits_N = [train_N, valid_N, test_N]

    spk_utts = dict((k, {}) for k in splits)
    # if command line specification is zero, it is equivalent to infinite
    spk_max_utts = {'train':np.inf if opts.max_train_utts_spk == 0 else opts.max_train_utts_spk,
                    'valid':np.inf if opts.max_valid_utts_spk == 0 else opts.max_valid_utts_spk,
                    'test':np.inf if opts.max_test_utts_spk == 0 else opts.max_test_utts_spk
                   }

    # 1) Train split, 2) Valid split, 3) Test split
    split_pointer = 0
    for si, (split, split_N) in enumerate(zip(splits, splits_N), start=1):
        TRAIN = (split == 'train')
        split_spks = spk_ids[split_pointer:split_pointer + split_N]
        total_wav_dur = 0

        timings = []
        beg_t = timeit.default_timer()
        for spk_i, spk_ in enumerate(split_spks, start=1):
            wavs = glob.glob(os.path.join(data_root, WAV_DIR, 
                                          'p' + spk_, '*.wav'))
            # Start utterance counter per spk
            if spk_ not in spk_utts[split]:
                spk_utts[split][spk_] = 0
            for wi, wav in enumerate(wavs):
                if spk_utts[split][spk_] >= spk_max_utts[split]:
                    # Skip utterance if this speaker already has maxed out
                    #print('Speaker {} already maxed out (MAX {}): '
                    #      '{}'.format(spk_, spk_max_utts, spk_utts[spk_]))
                    continue
                # Just count a new utterance in
                spk_utts[split][spk_] += 1
                x, rate = librosa.load(wav, sr=None)
                if x.shape[0] < opts.min_len:
                    # Ignore  too short seqs
                    print('Ignoring wav {} for len is {} < {}'.format(wav,
                                                                      x.shape[0],
                                                                      opts.min_len))
                    continue
                total_wav_dur += x.shape[0]
                bname = os.path.basename(wav)
                data_cfg[split]['data'].append(
                    {'filename':os.path.join(WAV_DIR,
                                             'p' + spk_,
                                             bname),
                     'spk':spk_}
                )
                if spk_ not in data_cfg[split]['speakers']:
                    data_cfg[split]['speakers'].append(spk_)
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            print('{}/{} processed spks for split {}/{} ({}) '
                  'mbtime: {:.3f} s'.format(spk_i, len(split_spks),
                                            si, len(splits),
                                            split, np.mean(timings)),
                 end='\r')
        print('')
        # write total wav dur
        data_cfg[split]['total_wav_dur'] = total_wav_dur
        split_pointer += split_N


    # Write final config file onto specified output path
    with open(cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, default=None)
    parser.add_argument('--cfg_file', type=str, default='vctk_data.cfg')
    parser.add_argument('--train_split', type=float, default=0.88)
    parser.add_argument('--valid_split', type=float, default=0.06)
    parser.add_argument('--min_len', type=int, default=16000)
    parser.add_argument('--max_train_utts_spk', type=int, default=0,
                        help='Maximum training utterances per spk. '
                             'This allows to make smaller trainset '
                             'to experiment with N utts per spk. '
                             'If it is zero, it takes all utts '
                             '(Def: 0).')
    parser.add_argument('--max_valid_utts_spk', type=int, default=0,
                        help='Maximum validation utterances per spk. '
                             'This allows to make smaller valset '
                             'to experiment with N utts per spk. '
                             'If it is zero, it takes all utts '
                             '(Def: 0).')

    parser.add_argument('--max_test_utts_spk', type=int, default=0,
                        help='Maximum test utterances per spk. '
                             'This allows to make smaller testset '
                             'to experiment with N utts per spk. '
                             'If it is zero, it takes all utts '
                             '(Def: 0).')

    opts = parser.parse_args()

    main(opts)
