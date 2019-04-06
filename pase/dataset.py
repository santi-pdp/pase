import torch
from torch.utils.data import Dataset
import soundfile as sf
import json
import librosa
import os
import random
import numpy as np
from collections import defaultdict


class DictCollater(object):

    def __init__(self, batching_keys=['chunk',
                                      'chunk_ctxt',
                                      'chunk_rand',
                                      'lps',
                                      'mfcc',
                                      'prosody'],
                labs=False):
        self.batching_keys = batching_keys
        self.labs = labs

    def __call__(self, batch):
        batches = {}
        lab_b = False
        labs = None
        lab_batches = []
        for sample in batch:
            if len(sample) > 1 and self.labs:
                labs = sample[1:]
                sample = sample[0]
                if len(lab_batches) == 0:
                    for lab in labs:
                        lab_batches.append([])
            for k, v in sample.items():
                if k not in self.batching_keys:
                    continue
                if k not in batches:
                    batches[k] = []
                if v.dim() == 1:
                    v = v.view(1, 1, -1)
                elif v.dim() == 2:
                    v = v.unsqueeze(0)
                else:
                    raise ValueError('Error in collating dimensions for size '
                                     '{}'.format(v.size()))
                batches[k].append(v)
            if labs is not None:
                for lab_i, lab in enumerate(labs):
                    lab_batches[lab_i].append(lab)
        for k in batches.keys():
            batches[k] = torch.cat(batches[k], dim=0)
        if labs is not None:
            rets = [batches]
            for li in range(len(lab_batches)):
                rets.append(torch.tensor(lab_batches[li]))
        else:
            rets = batches
        return rets

def uttwav_collater(batch):
    """ Simple collater where (wav, utt) pairs are
    given by the a dataset, and (wavs, utts, lens) are
    returned
    """
    max_len = 0
    for sample in batch:
        wav, uttname = sample
        if wav.shape[0] > max_len:
            max_len = wav.shape[0]

    wavs = []
    utts = []
    lens = []

    for sample in batch:
        wav, uttname = sample
        T = wav.shape[0]
        P = max_len - T
        if P > 0:
            wav = np.concatenate((wav,
                                  np.zeros((P,))),
                                 axis=0)
        wavs.append(wav)
        utts.append(uttname)
        lens.append(T)
    return torch.FloatTensor(wavs), utts, torch.LongTensor(lens)

class WavDataset(Dataset):

    def __init__(self, data_root, data_cfg_file, split, 
                 transform=None, sr=None, return_uttname=False,
                 return_spk=False,
                 verbose=True):
        # sr: sampling rate, (Def: None, the one in the wav header)
        self.sr = sr
        self.data_root = data_root
        self.data_cfg_file = data_cfg_file
        if not isinstance(data_cfg_file, str):
            raise ValueError('Please specify a path to a cfg '
                             'file for loading data.')

        self.return_uttname = return_uttname
        self.return_spk = return_spk
        self.split = split
        self.transform = transform
        with open(data_cfg_file, 'r') as data_cfg_f:
            self.data_cfg = json.load(data_cfg_f)
            self.spk_info = self.data_cfg['speakers']
            if verbose:
                print('Found {} speakers info'.format(len(self.spk_info)))
                wavs = self.data_cfg[split]['data']
                print('Found {} files in {} split'.format(len(wavs),
                                                          split))
                spks = self.data_cfg[split]['speakers']
                print('Found {} speakers in {} split'.format(len(spks),
                                                             split))
                self.total_wav_dur = self.data_cfg[split]['total_wav_dur']
                if 'spk2idx' in self.data_cfg and return_spk:
                    self.spk2idx = self.data_cfg['spk2idx']
                    print('Loaded spk2idx with {} '
                          'speakers'.format(len(self.spk2idx)))
            self.wavs = wavs
        self.wav_cache = {}

    def __len__(self):
        return len(self.wavs)

    def retrieve_cache(self, fname, cache):
        if fname in cache:
            return cache[fname]
        else:
            wav, rate = librosa.load(fname, sr=self.sr)
            cache[fname] = wav
            return wav

    def __getitem__(self, index):
        uttname = self.wavs[index]['filename']
        wname = os.path.join(self.data_root, uttname)
        wav = self.retrieve_cache(wname, self.wav_cache)
        if self.transform is not None:
            wav = self.transform(wav)
        rets = [wav]
        if self.return_uttname:
            rets = rets + [uttname]
        if self.return_spk:
            rets = rets + [self.spk2idx[self.wavs[index]['speaker']]]
        if len(rets) == 1: 
            return rets[0]
        else: 
            return rets

class PairWavDataset(WavDataset):
    """ Return paired wavs, one is current wav and the other one is a randomly
        chosen one.
    """
    def __init__(self, data_root, data_cfg_file, split, 
                 transform=None, sr=None, verbose=True):
        super().__init__(data_root, data_cfg_file, split, transform=transform, 
                         sr=sr,
                         verbose=verbose)
        self.rwav_cache = {}

    def __getitem__(self, index):
        # Here we select two wavs, the current one and a randomly chosen one
        wname = os.path.join(self.data_root, self.wavs[index]['filename'])
        wav = self.retrieve_cache(wname, self.wav_cache)
        # create candidate indices without current index
        indices = list(range(len(self.wavs)))
        indices.remove(index)
        rindex = random.choice(indices)
        rwname = os.path.join(self.data_root, self.wavs[rindex]['filename'])
        rwav = self.retrieve_cache(rwname, self.rwav_cache)
        if self.transform is not None:
            ret = self.transform({'raw': wav, 'raw_rand': rwav})
            return ret
        else:
            return wav, rwav

        

if __name__ == '__main__':
    print('WavDataset')
    print('-' * 30)
    dset = WavDataset('/veu/spascual/DB/VCTK', '../data/vctk_data.cfg', 'train')
    print(dset[0])
    print(dset[0].shape)
    print('=' * 30)

    dset = PairWavDataset('/veu/spascual/DB/VCTK', '../data/vctk_data.cfg', 'train')
    print('PairWavDataset')
    print('-' * 30)
    wav, rwav = dset[0]
    print('({}, {})'.format(wav.shape, rwav.shape))
    print('=' * 30)
