import argparse
from python_speech_features import fbank
import multiprocessing as mp
import numpy as np
import os
import tqdm
import soundfile as sf
import glob


def wav2fbank(args):
    wavname, out_dir, nfilt, log = args
    x, rate = sf.read(wavname)
    fb, egy = fbank(x, rate, nfilt=nfilt)
    if log:
        fb = np.log(fb)
    bname = os.path.splitext(os.path.basename(wavname))[0]
    outfile = os.path.join(out_dir, bname + '.fb')
    np.save(outfile, fb)

def main(opts):
    tasks = [(w, opts.out_dir, opts.nfilt, opts.log) for w in \
             glob.glob(os.path.join(opts.wav_dir, '*.wav'))]
    pool = mp.Pool(opts.num_workers)
    for _ in tqdm.tqdm(pool.imap_unordered(wav2fbank, tasks),
                       total=len(tasks)):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--nfilt', type=int, default=40)
    
    opts = parser.parse_args()
    if not os.path.exists(opts.out_dir):
        os.makedirs(opts.out_dir)
    main(opts)
