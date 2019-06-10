import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import soundfile as sf
import json
import tqdm
import librosa
import pickle
import os
import random
import numpy as np
from collections import defaultdict


class DictCollater(object):

    def __init__(self, batching_keys=['cchunk',
                                      'chunk',
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
                lab_batches_T = lab_batches[li]
                lab_batches_T = torch.tensor(lab_batches_T)
                rets.append(lab_batches_T)
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

def ft2spk_collater(batch):
    """ Simple collater where (fbank, spkid) pairs are
    given by the a dataset, and (fbanks, spkids, lens) are
    returned
    """
    max_len = 0
    for sample in batch:
        ft, _ = sample
        if ft.shape[1] > max_len:
            max_len = ft.shape[1]

    fts = []
    labs = []
    lens = []

    for sample in batch:
        ft, lab = sample
        seq_len = ft.shape[1]
        if seq_len < max_len:
            P = max_len - seq_len
            # repeat this amount at the beginning 
            rep = int(math.ceil(P / seq_len))
            if rep > 1:
                ft = torch.cat((ft.repeat(1, rep), ft), dim=1)
                ft = ft[:, -max_len:]
            else:
                pad = ft[:, :P]
                ft = torch.cat((pad, ft), dim=1)
        elif seq_len > max_len:
            # trim randomly within utterance
            idxs = list(range(seq_len - max_len))
            beg_i = random.choice(idxs)
            ft = ft[:, beg_i:]
        fts.append(ft.unsqueeze(0))
        labs.append(lab.unsqueeze(0))
        lens.append(seq_len)
    return torch.cat(fts, dim=0), torch.cat(labs, dim=0), lens

class WavDataset(Dataset):

    def __init__(self, data_root, data_cfg_file, split, 
                 transform=None, sr=None,
                 return_spk=False,
                 preload_wav=False,
                 return_uttname=False,
                 transforms_cache=None,
                 distortion_transforms=None,
                 whisper_folder=None,
                 noise_folder=None,
                 distortion_probability=0.4,
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
        self.transforms_cache = transforms_cache
        self.distortion_transforms = distortion_transforms
        self.whisper_folder = whisper_folder
        self.noise_folder = noise_folder
        self.distortion_probability = distortion_probability
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
        if whisper_folder is not None:
            self.whisper_cache = {}
        if noise_folder is not None:
            self.noise_cache = {}
        if preload_wav:
            print('Pre-loading wavs to memory')
            for wavstruct in tqdm.tqdm(self.wavs, total=len(self.wavs)):
                uttname = wavstruct['filename']
                wname = os.path.join(self.data_root, uttname)
                self.retrieve_cache(wname, self.wav_cache)
                if hasattr(self, 'whisper_cache'):
                    dwname = os.path.join(whisper_folder, uttname)
                    self.retrieve_cache(dwname, self.whisper_cache)
                if hasattr(self, 'noise_cache'):
                    nwname = os.path.join(noise_folder, uttname)
                    self.retrieve_cache(nwname, self.noise_cache)


    def __len__(self):
        return len(self.wavs)

    def retrieve_cache(self, fname, cache):
        if fname in cache:
            return cache[fname]
        else:
            wav, rate = sf.read(fname)
            wav = wav.astype(np.float32)
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
                 transform=None, sr=None, verbose=True,
                 return_uttname=False,
                 transforms_cache=None,
                 distortion_transforms=None,
                 whisper_folder=None,
                 noise_folder=None,
                 distortion_probability=0.4,
                 preload_wav=False):
        super().__init__(data_root, data_cfg_file, split, transform=transform, 
                         sr=sr, preload_wav=preload_wav,
                         return_uttname=return_uttname,
                         transforms_cache=transforms_cache,
                         distortion_transforms=distortion_transforms,
                         whisper_folder=whisper_folder,
                         noise_folder=noise_folder,
                         distortion_probability=distortion_probability,
                         verbose=verbose)
        self.rwav_cache = {}

    def __getitem__(self, index):
        uttname = self.wavs[index]['filename']
        # Here we select two wavs, the current one and a randomly chosen one
        wname = os.path.join(self.data_root, uttname)
        wav = self.retrieve_cache(wname, self.wav_cache)
        # create candidate indices without current index
        indices = list(range(len(self.wavs)))
        indices.remove(index)
        rindex = random.choice(indices)
        rwname = os.path.join(self.data_root, self.wavs[rindex]['filename'])
        rwav = self.retrieve_cache(rwname, self.rwav_cache)
        pkg = {'raw': wav, 'raw_rand': rwav,
               'uttname':uttname, 'split':self.split}
        # Apply the set of 'target' transforms on the clean data
        if self.transform is not None:
            pkg = self.transform(pkg)
        do_addnoise = False
        do_whisper = False
        # Then select possibly a distorted version of the 'current' chunk
        if hasattr(self, 'whisper_cache'):
            do_whisper = random.random() <= self.distortion_probability
            if do_whisper:
                dwname = os.path.join(self.whisper_folder, uttname)
                #print('getting whisper file: ', dwname)
                dwav = self.retrieve_cache(dwname, 
                                           self.whisper_cache)
                pkg['raw'] = torch.tensor(dwav)
        # Check if additive noise to be added
        if hasattr(self, 'noise_cache'):
            do_addnoise = random.random() <= self.distortion_probability
            if do_addnoise:
                nwname = os.path.join(self.noise_folder, uttname)
                #print('getting noise file: ', nwname)
                noise = self.retrieve_cache(nwname,
                                            self.noise_cache)
                if noise.shape[0] < pkg['raw'].size(0):
                    P_ = pkg['raw'].size(0) - noise.shape[0]
                    noise_piece = noise[-P_:][::-1]
                    noise = np.concatenate((noise, noise_piece), axis=0)
                noise = torch.FloatTensor(noise)
                pkg['raw'] = pkg['raw'] + noise

        pkg['cchunk'] = pkg['chunk'].squeeze(0)

        if do_addnoise or do_whisper:
            # re-chunk raw into chunk if boundaries available in pkg
            if 'chunk_beg_i' in pkg and 'chunk_end_i' in pkg:
                beg_i = pkg['chunk_beg_i']
                end_i = pkg['chunk_end_i']
                # separate clean chunk version
                # make distorted chunk
                pkg['chunk'] = pkg['raw'][beg_i:end_i]

        if self.distortion_transforms:
            pkg = self.distortion_transforms(pkg)

        if self.transform is None:
            # if no transforms happened do not send a package
            return pkg['chunk'], pkg['raw_rand']
        else:
            # otherwise return the full package
            return pkg

        
class FeatsClassDataset(Dataset):
    def __init__(self, data_root, utt2class, split_list, 
                 stats=None, verbose=True, ext='fb.npy'):
        self.data_root = data_root
        self.ext = ext
        if not isinstance(utt2class, str):
            raise ValueError('Please specify a path to a utt2class '
                             'file for loading data.')
        if not isinstance(split_list, str):
            raise ValueError('Please specify a path to a split_list '
                             'file for loading data.')
        utt2class_ext = utt2class.split('.')[1]
        if utt2class_ext == 'json':
            with open(utt2class, 'r') as u2s_f:
                self.utt2class = json.load(u2s_f)
        else:
            self.utt2class = np.load(utt2class)
            self.utt2class = dict(self.utt2class.any())
        print('Found {} speakers'.format(len(set(self.utt2class.values()))))
        with open(split_list, 'r') as sl_f:
            self.split_list = [l.rstrip() for l in sl_f]
            print('Found {} fbank files'.format(len(self.split_list)))
        if stats is not None:
            with open(stats, 'rb') as stats_f:
                self.stats = pickle.load(stats_f)

    def __len__(self):
        return len(self.split_list)

    def z_norm(self, x):
        assert hasattr(self, 'stats')
        stats = self.stats
        mean = torch.FloatTensor(stats['mean']).view(-1, 1)
        std = torch.FloatTensor(stats['std']).view(-1, 1)
        x = (x - mean) / std
        return x

    def __getitem__(self, index):
        item = self.split_list[index]
        bname = os.path.splitext(item)[0]
        ft_file = os.path.join(self.data_root, bname + '.' + self.ext)
        ft = torch.FloatTensor(np.load(ft_file).T)
        if hasattr(self, 'stats'):
            ft = self.z_norm(ft)
        seq_len = ft.shape[1]
        spk_id = self.utt2class[item]
        return ft, torch.LongTensor([spk_id])

class WavClassDataset(Dataset):
    """ Simple Wav -> classID dataset """
    def __init__(self, data_root, utt2class, split_list, 
                 chunker=None, 
                 verbose=True):
        self.data_root = data_root
        if not isinstance(utt2class, str):
            raise ValueError('Please specify a path to a utt2class '
                             'file for loading data.')
        if not isinstance(split_list, str) and not isinstance(split_list, list):
            raise ValueError('Please specify a path to a split_list '
                             'file for loading data or to the list itself.')
        utt2class_ext = utt2class.split('.')[1]
        if utt2class_ext == 'json':
            with open(utt2class, 'r') as u2s_f:
                self.utt2class = json.load(u2s_f)
        else:
            self.utt2class = np.load(utt2class)
            self.utt2class = dict(self.utt2class.any())
        print('Found {} classes'.format(len(set(self.utt2class.values()))))
        self.chunker = chunker
        if isinstance(split_list, list):
            self.split_list = split_list
        else:
            with open(split_list, 'r') as sl_f:
                self.split_list = [l.rstrip() for l in sl_f]
                print('Found {} wav files'.format(len(self.split_list)))

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, index):
        item = self.split_list[index]
        bname = os.path.splitext(item)[0]
        wav_file = os.path.join(self.data_root, bname + '.wav')
        wav, rate = sf.read(wav_file)
        wav = torch.FloatTensor(wav)
        if self.chunker is not None:
            if len(wav) < self.chunker.chunk_size + 1:
                P = self.chunker.chunk_size  + 1 - len(wav)
                wav = torch.cat((wav, 
                                 torch.zeros(P)),
                                dim=0)
            wav = self.chunker(wav)
            wav = wav['chunk']
        spk_id = self.utt2class[item]
        return wav, torch.LongTensor([spk_id])

if __name__ == '__main__':
    """
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
    """
    from torch.utils.data import DataLoader
    dset = FbankSpkDataset('../data/LibriSpeech/Fbanks24', 
                           '../data/LibriSpeech/libri_dict.npy',
                           '../data/LibriSpeech/libri_tr.scp')
    x, y = dset[0]
    print(x.shape)
    print(y.shape)
    dloader = DataLoader(dset, batch_size=10, shuffle=True,
                         collate_fn=ft2spk_collater)
    x = next(dloader.__iter__())
    print(x[0].shape)
    print(x[1].shape)
    print(x[2])
