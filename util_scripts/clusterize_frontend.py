from sklearn.cluster import KMeans
from pase.models.frontend import wf_builder
from pase.dataset import PairWavDataset, DictCollater
from torchvision.transforms import Compose
from pase.transforms import *
from torch.utils.data import DataLoader
import numpy as np
import argparse
import timeit
import pickle
import os
import json


def cluster(opts):
    CUDA = True if torch.cuda.is_available() else False
    device = 'cuda' if CUDA else 'cpu'
    num_devices = 1
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if CUDA:
        torch.cuda.manual_seed_all(opts.seed)
        num_devices = torch.cuda.device_count()
        print('[*] Using CUDA {} devices'.format(num_devices))
    else:
        print('[!] Using CPU')
    fe = wf_builder(opts.fe_cfg)
    if opts.fe_ckpt is not None:
        fe.load_pretrained(opts.fe_ckpt, load_last=True, verbose=True)
    else:
        print('WARNING: No pretrained ckpt loaded for FE! Random clustering?')
    fe.to(device)
    fe.eval()
    trans = Compose([ToTensor(),
                     SingleChunkWav(opts.chunk_size, random_scale=False)])
    # Build Dataset(s) and DataLoader(s)
    dset = PairWavDataset(opts.data_root, opts.data_cfg, 'train',
                         transform=trans)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, collate_fn=DictCollater(),
                         num_workers=opts.num_workers)
    # acumulate train chunks and do clustering on them,
    # with each chunk containing several frames
    X = []
    timings = []
    N = opts.num_samples // opts.batch_size
    beg_t = timeit.default_timer()
    for bidx in range(1, N + 1, 1):
        batch = next(dloader.__iter__())
        chunk = batch['chunk']
        y = fe(chunk.to(device)).mean(dim=2)
        X.append(y.view(-1, y.size(-1)).cpu().data.numpy())
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        if bidx % opts.log_freq == 0 or bidx >= N:
            print('Forwarded batch {:4d}/{:4d}, btime: {:.2f} s, '
                  'mbtime: {:.2f} s'.format(bidx, N, timings[-1],
                                            np.mean(timings)),
                  end='\r')
    print()
    X = np.concatenate(X, axis=0)
    print('Total X shape: ', X.shape)
    print('Running KMeans...')
    beg_t = timeit.default_timer()
    kmeans = KMeans(n_clusters=opts.k_clusters, n_jobs=opts.n_jobs,
                    verbose=0).fit(X)
    end_t = timeit.default_timer()
    print('Clusterized in {:.2f} s'.format(end_t - beg_t))
    print('Saving KMeans...')
    with open(os.path.join(opts.save_path, 'kmeans.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    print('Finished program')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_cfg', type=str, 
                        default='data/librispeech_data.cfg')
    parser.add_argument('--data_root', type=str, 
                        default='data/LibriSpeech/Librispeech_spkid_sel')
    parser.add_argument('--fe_cfg', type=str, default=None)
    parser.add_argument('--fe_ckpt', type=str, default=None)
    parser.add_argument('--chunk_size', type=int, default=16000)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--log_freq', type=int, default=15)
    parser.add_argument('--k_clusters', type=int, default=128,
                        help='Number of clusters (Def: 128).')
    parser.add_argument('--save_path', type=str, default='kmeans_FE')
    opts = parser.parse_args()
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    with open(os.path.join(opts.save_path, 'cluster.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))
    cluster(opts)

