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
                         'TFCommands/train/audio is located, containing '
                         'its commands sub-dirs.')
    # First read the amount of speakers in the whole commands dataset
    commands = os.listdir(opts.data_root)
    spks = {}
    spk2idx = {}
    spk2numutts = {}
    spk2utt = {}
    utt2dur = {}
    for command in commands:
        if '_background_noise_' in command:
            continue
        spks[command] = {}
        wavs = glob.glob(os.path.join(opts.data_root, command, '*.wav'))
        for wav in wavs:
            # filter by min duration each utterance
            x, sr = librosa.load(wav, sr=None)
            dur = x.shape[0]
            if dur < opts.min_len:
                print('Skipping {} for dur {} < {}'.format(wav,
                                                           dur,
                                                           opts.min_len))
                continue
            utt = os.path.join(command, os.path.basename(wav))
            utt2dur[utt] = dur
            spkid = os.path.basename(wav).split('_')[0]
            if spkid not in spks[command]:
                spks[command][spkid] = 0
            spks[command][spkid] += 1
            if spkid not in spk2idx:
                spk2idx[spkid] = len(spk2idx) + 1
            if spkid not in spk2numutts:
                spk2numutts[spkid] = 0
                spk2utt[spkid] = []
            spk2numutts[spkid] += 1
            spk2utt[spkid].append(utt)
    MIN_FILES = opts.min_commands
    # filtered speaker IDs
    fspks = {}
    for spk, utts in spk2numutts.items():
        if utts >= MIN_FILES:
            fspks[spk] = utts

    # We have the speakers cfg section, now let's build the data split by spks
    spk_ids = list(fspks.keys())
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
                'all':{'data':[],
                       'speakers':[]},
                'speakers':spks}

    splits = ['train', 'valid', 'test']
    splits_N = [train_N, valid_N, test_N]

    # 1) Train split, 2) Valid split, 3) Test split
    split_pointer = 0
    for si, (split, split_N) in enumerate(zip(splits, splits_N), start=1):
        split_spks = spk_ids[split_pointer:split_pointer + split_N]
        total_wav_dur = 0

        timings = []
        beg_t = timeit.default_timer()
        for spk_i, spk_ in enumerate(split_spks, start=1):
            wavs = spk2utt[spk_]
            #wavs = glob.glob(os.path.join(data_root, WAV_DIR, 
            #                              'p' + spk_, '*.wav'))
            for wi, wav in enumerate(wavs):
                dur = utt2dur[wav]
                total_wav_dur += dur
                bname = os.path.basename(wav)
                data_cfg[split]['data'].append(
                    {'filename':wav,
                     'spk':spk_}
                )
                data_cfg['all']['data'].append(
                    {'filename':wav,
                     'spk':spk_}
                )
                if spk_ not in data_cfg[split]['speakers']:
                    data_cfg[split]['speakers'].append(spk_)
                if spk_ not in data_cfg['all']['speakers']:
                    data_cfg['all']['speakers'].append(spk_)
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

    total_split_durs = 0
    for split in splits:
        total_split_durs += data_cfg[split]['total_wav_dur']
    data_cfg['all']['total_wav_dur'] = total_split_durs

    # Write final config file onto specified output path
    with open(opts.cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, default=None)
    parser.add_argument('--min_commands', type=int, default=5,
                        help='Min number of commands for a spk '
                             'to be selected (Def: 5).')
    parser.add_argument('--cfg_file', type=str, default='tfcommands_data.cfg')
    parser.add_argument('--train_split', type=float, default=0.88)
    parser.add_argument('--valid_split', type=float, default=0.06)
    parser.add_argument('--min_len', type=int, default=8000)

    opts = parser.parse_args()

    main(opts)
