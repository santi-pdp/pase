import argparse
from ahoproc_tools.io import read_aco_file, write_aco_file
from waveminionet.transforms import *
import torch
import numpy as np
import soundfile as sf
import timeit
import glob
import os


def main(opts):
    acos = glob.glob(os.path.join(opts.aco_dir, '*.mfcc'))
    wavs = glob.glob(os.path.join(opts.wav_dir, '*.wav'))
    print('num acos found: ', len(acos))
    print('num wavs found: ', len(wavs))
    assert len(acos) == len(wavs), '{} != {}'.format(len(acos),
                                                     len(wavs))
    ptrans = Prosody(hop=opts.hop, win=opts.win)
    timings = []
    beg_t = timeit.default_timer()
    for ai, aco in enumerate(acos, start=1):
        bname = os.path.splitext(os.path.basename(aco))[0]
        wav = os.path.join(opts.wav_dir, bname + '.wav')
        # read aco file
        aco_data = read_aco_file(aco, (-1, opts.aco_order))
        x, rate = sf.read(wav)
        # TODO: extract prosodic features
        proso = ptrans({'chunk':torch.FloatTensor(x)})['prosody']
        proso = proso.data.numpy().T
        aco_data = aco_data[:proso.shape[0], :]
        composed = np.concatenate((aco_data,
                                   proso), axis=1)
        write_aco_file(os.path.join(opts.out_dir, bname + ".proso"),
                       composed)
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        if ai % 100 == 0 or ai >= len(acos):
            print('Processed file {:4d}/{:4d}, mfile_time: {:.2f} s'
                  ''.format(ai, len(acos), np.mean(timings)),
                  end='\r')
    print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aco_dir', type=str, default=None)
    parser.add_argument('--wav_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--aco_order', type=int, default=39)
    parser.add_argument('--hop', type=int, default=160)
    parser.add_argument('--win', type=int, default=400)

    opts = parser.parse_args()

    if not os.path.exists(opts.out_dir):
        os.makedirs(opts.out_dir)

    main(opts)
