import torch
from torch.utils.data import DataLoader
from pase.dataset import PairWavDataset, DictCollater, MetaWavConcatDataset
from torchvision.transforms import Compose
from pase.transforms import *
import argparse
import pickle
from train import make_transforms
import pase
from pase.utils import *

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

    minions_cfg = worker_parser(opts.net_cfg)
    trans, batch_keys = make_transforms(opts.chunk_size, minions_cfg,
                                        opts.hop_size)
    """
    trans = Compose([
        ToTensor(),
        MIChunkWav(opts.chunk_size),
        #LPS(hop=opts.hop_size, win=opts.win_size),
        #Gammatone(hop=opts.hop_size),
        #LPC(hop=opts.hop_size),
        #FBanks(hop=opts.hop_size),
        #MFCC(hop=opts.hop_size, win=opts.win_size),
        #KaldiMFCC(kaldi_root=opts.kaldi_root, hop=opts.hop_size, win=opts.win_size),
        #KaldiPLP(kaldi_root=opts.kaldi_root, hop=opts.hop_size, win=opts.win_size),
        #Prosody(hop=opts.hop_size)
        LPS(hop=opts.LPS_hop,win=opts.LPS_win,der_order=opts.LPS_der_order),
        Gammatone(hop=opts.gammatone_hop,win=opts.gammatone_win,der_order=opts.gammatone_der_order),
        #LPC(hop=opts.LPC_hop),
        FBanks(hop=opts.fbanks_hop,win=opts.fbanks_win,der_order=opts.fbanks_der_order),
        MFCC(hop=opts.mfccs_hop,win=opts.mfccs_win,order=opts.mfccs_order,der_order=opts.mfccs_der_order),
        #MFCC_librosa(hop=opts.mfccs_librosa_hop,win=opts.mfccs_librosa_win,order=opts.mfccs_librosa_order,der_order=opts.mfccs_librosa_der_order,n_mels=opts.mfccs_librosa_n_mels,htk=opts.mfccs_librosa_htk),
        #KaldiMFCC(kaldi_root=opts.kaldi_root, hop=opts.kaldimfccs_hop, win=opts.kaldimfccs_win,num_mel_bins=opts.kaldimfccs_num_mel_bins,num_ceps=opts.kaldimfccs_num_ceps,der_order=opts.kaldimfccs_der_order),
        #KaldiPLP(kaldi_root=opts.kaldi_root, hop=opts.kaldiplp_hop, win=opts.kaldiplp_win),
        Prosody(hop=opts.prosody_hop, win=opts.prosody_win, der_order=opts.prosody_der_order)
    ])
    """

    dsets = []
    for idx in range(len(opts.data_root)):
        dataset = getattr(pase.dataset, opts.dataset[idx])
        dset = dataset(opts.data_root[idx], opts.data_cfg[idx], 'train',
                       transform=trans, ihm2sdm=opts.ihm2sdm)
        #dset = PairWavDataset(opts.data_root[idx], opts.data_cfg[idx], 'train',
        #                 transform=trans)
        dsets.append(dset)

    if len(dsets) > 1:
        return MetaWavConcatDataset(dsets), batch_keys
    else:
        return dsets[0], batch_keys

def extract_stats(opts):
    dset = build_dataset_providers(opts)
    collater_keys = dset[-1]
    dset = dset[0]
    collater = DictCollater()
    collater.batching_keys.extend(collater_keys)
    dloader = DataLoader(dset, batch_size = 100,
                         shuffle=True, collate_fn=collater,
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
    #parser.add_argument('--win_size', type=int, default=400)
    
    # setting hop/wlen for each features
    parser.add_argument('--LPS_hop', type=int, default=160)
    parser.add_argument('--LPS_win', type=int, default=400)
    parser.add_argument('--LPS_der_order', type=int, default=0)
    #parser.add_argument('--gammatone_hop', type=int, default=160)
    parser.add_argument('--gammatone_win', type=int, default=400)
    parser.add_argument('--gammatone_der_order', type=int, default=0)
    #parser.add_argument('--LPC_hop', type=int, default=160)
    parser.add_argument('--LPC_win', type=int, default=400)
    #parser.add_argument('--fbanks_hop', type=int, default=160)
    parser.add_argument('--fbanks_win', type=int, default=400)
    parser.add_argument('--fbanks_der_order', type=int, default=0)
    #parser.add_argument('--mfccs_hop', type=int, default=160)
    parser.add_argument('--mfccs_win', type=int, default=400)
    parser.add_argument('--mfccs_order', type=int, default=20)
    parser.add_argument('--mfccs_der_order', type=int, default=0)
    #parser.add_argument('--prosody_hop', type=int, default=160)
    parser.add_argument('--prosody_win', type=int, default=400)
    parser.add_argument('--prosody_der_order', type=int, default=0)
    #parser.add_argument('--kaldimfccs_hop', type=int, default=160)
    parser.add_argument('--kaldimfccs_win', type=int, default=400)
    parser.add_argument('--kaldimfccs_der_order', type=int, default=0)
    parser.add_argument('--kaldimfccs_num_mel_bins', type=int, default=20)
    parser.add_argument('--kaldimfccs_num_ceps', type=int, default=20)
    #parser.add_argument('--kaldiplp_hop', type=int, default=160)
    parser.add_argument('--kaldiplp_win', type=int, default=400)
    
    #parser.add_argument('--mfccs_librosa_hop', type=int, default=160)
    parser.add_argument('--mfccs_librosa_win', type=int, default=400)
    parser.add_argument('--mfccs_librosa_order', type=int, default=20)
    parser.add_argument('--mfccs_librosa_der_order', type=int, default=0)
    parser.add_argument('--mfccs_librosa_n_mels', type=int, default=40)
    parser.add_argument('--mfccs_librosa_htk', type=int, default=True)
    parser.add_argument('--net_cfg', type=str, default=None)

    
    parser.add_argument('--ihm2sdm', type=str, default=None,
                        help='Relevant only to ami-like dataset providers')
    parser.add_argument('--kaldi_root', type=str, default=None,
                        help='Absolute path to kaldi installation. Possibly of use for feature related bits.')
    opts = parser.parse_args()
    extract_stats(opts)
