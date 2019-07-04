import arff
import numpy as np
from ahoproc_tools.interpolate import interpolation
import tqdm
import argparse
import pickle
import os


def main(opts):
    for ai, afile in tqdm.tqdm(enumerate(opts.arff_files), total=len(opts.arff_files)):
        with open(afile) as af:
            data = arff.load(af)
            attrs = [at[0] for at in data['attributes']]
            f0_idx = attrs.index('F0_sma')
            data = data['data']
            array = []
            X = []
            for dpoint in data:
                # ignore name, timestamp and class
                f0_val = dpoint[f0_idx]
                if f0_val > 0:
                    dpoint[f0_idx] = np.log(f0_val)
                else:
                    dpoint[f0_idx] = -1e10
                array.append(dpoint[2:-1])
            array = np.array(array, dtype=np.float32)
            lf0, _ = interpolation(array[:, -1], -1e10)
            array[:, -1] = lf0
            if opts.out_stats is not None:
                X.append(array)
            npfile = os.path.splitext(afile)[0]
            np.save(os.path.join(npfile), array.T)
    if opts.out_stats is not None:
        X = np.concatenate(X, axis=0)
        mn = np.mean(X, axis=0)
        sd = np.std(X, axis=0)
        with open(opts.out_stats, 'wb') as out_f:
            pickle.dump({'mean':mn, 'std':sd}, out_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arff_root', type=str,
                        default='data/IEMOCAP_ahsn_leave-two-speaker-out_LLD')
    parser.add_argument('--arff_files', type=str, default=None, nargs='+')
    parser.add_argument('--out_stats', type=str, default=None)

    opts = parser.parse_args()
    main(opts)

