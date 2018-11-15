import torch
from torch.utils.data import Dataset, DataLoader
import json
import librosa
import os
import random


class WavDataset(Dataset):

    def __init__(self, data_root, data_cfg_file, split, sr=None,
                 verbose=True):
        # sr: sampling rate, (Def: None, the one in the wav header)
        self.sr = sr
        self.data_root = data_root
        self.data_cfg_file = data_cfg_file
        if not isinstance(data_cfg_file, str):
            raise ValueError('Please specify a path to a cfg '
                             'file for loading data.')

        self.split = split
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
            self.wavs = wavs

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, index):
        wname = os.path.join(self.data_root, self.wavs[index]['filename'])
        wav, rate = librosa.load(wname, sr=self.sr)
        return wav

class PairWavDataset(WavDataset):
    """ Return paired wavs, one is current wav and the other one is a randomly
        chosen one.
    """
    def __init__(self, data_root, data_cfg_file, split, sr=None, verbose=True):
        super().__init__(data_root, data_cfg_file, split, sr=sr,
                         verbose=verbose)


    def __getitem__(self, index):
        # Here we select two wavs, the current one and a randomly chosen one
        wname = os.path.join(self.data_root, self.wavs[index]['filename'])
        wav, rate = librosa.load(wname, sr=self.sr)
        # create candidate indices without current index
        indices = list(range(len(self.wavs)))
        indices.remove(index)
        rindex = random.choice(indices)
        rwname = os.path.join(self.data_root, self.wavs[rindex]['filename'])
        rwav, rrate = librosa.load(rwname, sr=self.sr)
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
