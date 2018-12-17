import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import json
import timeit
import glob
import os


def load_train_files(root_path, cfg, split):
    spk2idx = {}
    npys = cfg[split]['wav_files']
    labs = cfg[split]['spk_ids']
    Y = []
    X = []
    spk2idx = {}
    for npy, lab in zip(npys, labs):
        npy_name = os.path.join(root_path, npy)
        x = np.load(npy_name)
        if lab not in spk2idx:
            spk2idx[lab] = len(spk2idx)
        X.append(x.T)
        Y += [spk2idx[lab]] * x.T.shape[0]
    return np.concatenate(X, axis=0), np.array(Y), spk2idx

def load_test_files(root_path, cfg):
    spk2idx = {}
    npys = cfg['test']['wav_files']
    labs = cfg['test']['spk_ids']
    Y = []
    X = []
    for npy, lab in zip(npys, labs):
        npy_name = os.path.join(root_path, npy)
        x = np.load(npy_name)
        if lab not in spk2idx:
            spk2idx[lab] = len(spk2idx)
        X.append(x.T)
        Y += [spk2idx[lab]]
    return X, Y

def main(opts):
    # find npy files in data dir
    with open(opts.data_cfg, 'r') as cfg_f:
        # contains train and test files
        cfg = json.load(cfg_f)
        train_X, train_Y, spk2idx = load_train_files(opts.data_root,
                                                     cfg, 'train')
        test_X, test_Y = load_test_files(opts.data_root, cfg)
        print('Loaded trainX: ', train_X.shape)
        print('Loaded trainY: ', train_Y.shape)
        neigh = KNeighborsClassifier(n_neighbors=opts.k, n_jobs=opts.n_jobs)
        neigh.fit(train_X, train_Y) 
        accs = []
        timings = []
        beg_t = timeit.default_timer()
        for te_idx in range(len(test_X)):
            test_x = test_X[te_idx]
            facc = []
            preds = [0.] * len(spk2idx)
            Y_ = neigh.predict(test_x)
            for ii in range(len(Y_)):
                preds[Y_[ii]] += 1
            y_ = np.argmax(preds, axis=0)
            y = test_Y[te_idx]
            if y_ == y:
                accs.append(1)
            else:
                accs.append(0.)
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            print('Processing test utterance {}/{}, muttime: {:.3f} s'
                  ''.format(te_idx + 1,
                            len(test_X),
                            np.mean(timings)))
        print('Score on {} samples: {}'.format(len(accs),
                                               np.mean(accs)))
        with open(opts.out_log, 'w') as out_f:
            out_f.write('{:.4f}'.format(np.asscalar(np.mean(accs))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=None)
    parser.add_argument('--data_cfg', type=str,
                        default=None,
                        help='Dictionary containing paths to train and '
                             'test data files.')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_jobs', type=int, default=None)
    parser.add_argument('--out_log', type=str, default=None)


    opts = parser.parse_args()
    assert opts.data_root is not None
    assert opts.data_cfg is not None
    assert opts.out_log is not None
    main(opts)
