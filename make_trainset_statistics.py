import torch
from torch.utils.data import DataLoader
from pase.dataset import PairWavDataset, DictCollater, MetaWavConcatDataset
from torchvision.transforms import Compose
from pase.transforms import *
import argparse
import pickle
import pase

def build_dataset_providers(opts):

    assert len(opts.data_root) > 0, (
        "Expected at least one data_root argument"
    )

    assert len(opts.data_root) == len(opts.data_cfg), (
        "Provide same number of data_root and data_cfg arguments"
    )

    if len(opts.data_root) == 1 and \
        len(opts.dataset) < 1:
        opts.dataset.append('PairWavDataset')

    assert len(opts.data_root) == len(opts.dataset), (
        "Provide same number of data_root and dataset arguments"
    )

    trans = Compose([
        ToTensor(),
        MIChunkWav(opts.chunk_size),
        LPS(hop=opts.hop_size),
        Gammatone(hop=opts.hop_size),
        #LPC(hop=opts.hop_size),
        FBanks(hop=opts.hop_size),
        MFCC(hop=opts.hop_size),
        Prosody(hop=opts.hop_size)
    ])

    dsets = []
    for idx in range(len(opts.data_root)):
        dataset = getattr(pase.dataset, opts.dataset[idx])
        dset = dataset(opts.data_root[idx], opts.data_cfg[idx], 'train',
                       transform=trans, ihm2sdm=opts.ihm2sdm)
        #dset = PairWavDataset(opts.data_root[idx], opts.data_cfg[idx], 'train',
        #                 transform=trans)
        dsets.append(dset)

    if len(dsets) > 1:
        return MetaWavConcatDataset(dsets)
    else:
        return dsets[0]

def extract_stats(opts):
    dset = build_dataset_providers(opts)
    dloader = DataLoader(dset, batch_size = 100,
                         shuffle=True, collate_fn=DictCollater(),
                         num_workers=opts.num_workers)
    # Compute estimation of bpe. As we sample chunks randomly, we
    # should say that an epoch happened after seeing at least as many
    # chunks as total_train_wav_dur // chunk_size
    bpe = (dset.total_wav_dur // opts.chunk_size) // 500
    data = {}
    # run one epoch of training data to extract z-stats of minions
    for bidx, batch in enumerate(dloader, start=1):
        print('Bidx: {}/{}'.format(bidx, bpe))
        for k, v in batch.items():
            if k in opts.exclude_keys:
                continue
            if k not in data:
                data[k] = []
            data[k].append(v)

        if bidx >= opts.max_batches:
            break

    stats = {}
    data = dict((k, torch.cat(v)) for k, v in data.items())
    for k, v in data.items():
        stats[k] = {'mean':torch.mean(torch.mean(v, dim=2), dim=0),
                    'std':torch.std(torch.std(v, dim=2), dim=0)}
    with open(opts.out_file, 'wb') as stats_f:
        pickle.dump(stats, stats_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', action='append', 
                        default=[])
    parser.add_argument('--data_cfg', action='append', 
                        default=[])
    parser.add_argument('--dataset', action='append', 
                        default=[])
    parser.add_argument('--exclude_keys', type=str, nargs='+', 
                        default=['chunk', 'chunk_rand', 'chunk_ctxt'])
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=16000)
    parser.add_argument('--max_batches', type=int, default=20)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--hop_size', type=int, default=160)
    parser.add_argument('--ihm2sdm', type=str, default=None,
                        help='Relevant only to ami-like dataset providers')

    opts = parser.parse_args()
    extract_stats(opts)
