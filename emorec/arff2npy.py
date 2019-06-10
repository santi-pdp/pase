import arff
import numpy as np
import tqdm
import argparse
import os


def main(opts):
    for afile in tqdm.tqdm(opts.arff_files, total=len(opts.arff_files)):
        with open(afile) as af:
            data = arff.load(af)
            attrs = [at[0] for at in data['attributes']]
            f0_idx = attrs.index('F0_sma')
            data = data['data']
            array = []
            for dpoint in data:
                # ignore name, timestamp and class
                f0_val = dpoint[f0_idx]
                if f0_val > 0:
                    dpoint[f0_idx] = np.log(f0_val)
                else:
                    dpoint[f0_idx] = -1e10
                array.append(dpoint[2:-1])
            array = np.array(array, dtype=np.float32)
            npfile = os.path.splitext(afile)[0]
            np.save(os.path.join(npfile), array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arff_root', type=str,
                        default='data/IEMOCAP_ahsn_leave-two-speaker-out_LLD')
    parser.add_argument('--arff_files', type=str, default=None, nargs='+')

    opts = parser.parse_args()
    main(opts)

