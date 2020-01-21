import glob
import os
import multiprocessing as mp
from pase.transforms import *
import tqdm
import argparse

def process_codec(args):
    c2 = Codec2Buffer()
    infile, outdir = args
    bname = os.path.basename(infile)
    outpath = os.path.join(outdir, bname)
    x, rate = sf.read(infile)
    y = c2({'chunk':torch.tensor(x)})['chunk']
    sf.write(outpath, y, rate)

def main(opts):
    assert opts.num_workers > 0, opts.num_workers
    pool = mp.Pool(opts.num_workers)
    wavs = glob.glob(os.path.join(opts.input_dir, '*.wav'))
    args = [(wav, opts.output_dir) for wav in wavs]
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    for _ in tqdm.tqdm(pool.imap_unordered(process_codec, args),
                       total=len(args)):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, default=None)
    parser.add_argument('output_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)

    opts = parser.parse_args()


    main(opts)

