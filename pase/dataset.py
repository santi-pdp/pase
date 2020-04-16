import torch
import torch.nn.functional as F
import re
import glob
from torch.utils.data import Dataset, ConcatDataset
import math
import torchaudio
import json
import tqdm
import pickle
import os
try:
    from .utils import *
except ImportError:
    from utils import *
import random
import numpy as np
from collections import defaultdict


class DictCollater(object):

    def __init__(self, batching_keys=['cchunk',
                                      'chunk',
                                      'chunk_ctxt',
                                      'chunk_rand',
                                      'overlap',
                                      'lps',
                                      'lpc',
                                      'gtn',
                                      'fbank',
                                      'mfcc',
                                      'mfcc_librosa',
                                      'prosody',
                                      'kaldimfcc',
                                      'kaldiplp'],
                 meta_keys=[],
                 labs=False):
        self.batching_keys = batching_keys
        self.labs = labs
        self.meta_keys = meta_keys

    def __call__(self, batch):
        batches = {}
        lab_b = False
        labs = None
        lab_batches = []
        meta = {}
        for sample in batch:
            if len(sample) > 1 and self.labs:
                labs = sample[1:]
                sample = sample[0]
                if len(lab_batches) == 0:
                    for lab in labs:
                        lab_batches.append([])
            for k, v in sample.items():
                if k in self.meta_keys:
                    if k not in meta:
                        meta[k] = []
                    meta[k].append(v)
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
        rets = [batches]
        if labs is not None:
            for li in range(len(lab_batches)):
                lab_batches_T = lab_batches[li]
                lab_batches_T = torch.tensor(lab_batches_T)
                rets.append(lab_batches_T)
        if len(meta) > 0:
            rets.append(meta)
        if len(rets) == 1:
            return rets[0]
        else:
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
                 cache_on_load=False,
                 distortion_probability=0.4,
                 zero_speech_p=0,
                 zero_speech_transform=None,
                 verbose=True,
                 *args, **kwargs):
        # sr: sampling rate, (Def: None, the one in the wav header)
        self.sr = sr
        self.data_root = data_root
        self.cache_on_load = cache_on_load
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
        self.zero_speech_p = zero_speech_p
        self.zero_speech_transform = zero_speech_transform
        self.whisper_folder = whisper_folder
        self.noise_folder = noise_folder
        self.preload_wav = preload_wav
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
                self.total_wav_dur = int(self.data_cfg[split]['total_wav_dur'])
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
        if (self.cache_on_load or self.preload_wav) and fname in cache:
            return cache[fname]
        else:
            wav, rate = torchaudio.load(fname)
            wav = wav.numpy().squeeze()
            #fix in case wav is stereo, in which case
            #pick first channel only
            if wav.ndim > 1:
                wav = wav[:,0]
            wav = wav.astype(np.float32)
            if self.cache_on_load:
                cache[fname] = wav
            return wav

    def __getitem__(self, index):
        if sample_probable(self.zero_speech_p):
            wav = zerospeech(int(5 * 16e3))
            if self.zero_speech_transform is not None:
                wav = self.zero_speech_transform(wav)
        else:
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rwav_cache = {}

    def __getitem__(self, index):
        # create candidate indices for random other wavs without current index
        indices = list(range(len(self.wavs)))
        indices.remove(index)
        rindex = random.choice(indices)
        rwname = os.path.join(self.data_root, self.wavs[rindex]['filename'])
        rwav = self.retrieve_cache(rwname, self.wav_cache)
        # Load current wav or generate the zero-version
        if sample_probable(self.zero_speech_p):
            ZERO_SPEECH = True
            wav = zerospeech(int(5 * 16e3))
            uttname = 'zerospeech.wav'
        else:
            ZERO_SPEECH = False
            uttname = self.wavs[index]['filename']
            # Here we select two wavs, the current one and a randomly chosen one
            wname = os.path.join(self.data_root, uttname)
            wav = self.retrieve_cache(wname, self.wav_cache)
            #print ('Wav shape for {} is {}'.format(uttname, wav.shape))
        pkg = {'raw': wav, 'raw_rand': rwav,
               'uttname': uttname, 'split': self.split}
        # Apply the set of 'target' transforms on the clean data
        if self.transform is not None:
            pkg = self.transform(pkg)

        pkg['cchunk'] = pkg['chunk'].squeeze(0)
        # initialize overlap label
        if 'dec_resolution' in pkg:
            pkg['overlap'] = torch.zeros(len(pkg['chunk']) // pkg['dec_resolution']).float()
        else:
            pkg['overlap'] = torch.zeros(len(pkg['chunk'])).float()

        if self.distortion_transforms and not ZERO_SPEECH:
            pkg = self.distortion_transforms(pkg)
        
        if self.zero_speech_transform and ZERO_SPEECH:
            pkg = self.zero_speech_transform(pkg)

        if self.transform is None:
            # if no transforms happened do not send a package
            return pkg['chunk'], pkg['raw_rand']
        else:
            # otherwise return the full package
            return pkg


class GenhancementDataset(Dataset):
    """ Return the regular package with current (noisy) wav, 
        random neighbor wav (also noisy), and clean output
    """

    def __init__(self, data_root, data_cfg_file, split,
                 transform=None, sr=None,
                 return_spk=False,
                 preload_wav=False,
                 return_uttname=False,
                 transforms_cache=None,
                 distortion_transforms=None,
                 whisper_folder=None,
                 noise_folder=None,
                 cache_on_load=False,
                 distortion_probability=0.4,
                 zero_speech_p=0,
                 zero_speech_transform=None,
                 verbose=True,
                 *args, **kwargs):
        # TODO: half of these useless arguments should be removed in 
        # this dataset, but need to homogeneize the datasets or something
        super().__init__()
        # sr: sampling rate, (Def: None, the one in the wav header)
        self.sr = sr
        self.data_root = data_root
        self.data_cfg_file = data_cfg_file
        if not isinstance(data_cfg_file, str):
            raise ValueError('Please specify a path to a cfg '
                             'file for loading data.')

        self.split = split
        self.transform = transform
        with open(data_cfg_file, 'r') as data_cfg_f:
            self.data_cfg = json.load(data_cfg_f)
            self.spk_info = self.data_cfg['speakers']
            if verbose:
                print('Found {} speakers info'.format(len(self.spk_info)))
                noisy_wavs = self.data_cfg[split]['data']
                print('Found {} files in {} split'.format(len(noisy_wavs),
                                                          split))
                spks = self.data_cfg[split]['speakers']
                print('Found {} speakers in {} split'.format(len(spks),
                                                             split))
                self.total_wav_dur = int(self.data_cfg[split]['total_wav_dur'])
                if 'spk2idx' in self.data_cfg and return_spk:
                    self.spk2idx = self.data_cfg['spk2idx']
                    print('Loaded spk2idx with {} '
                          'speakers'.format(len(self.spk2idx)))
            self.noisy_wavs = noisy_wavs

    def __len__(self):
        return len(self.noisy_wavs)

    def __getitem__(self, index):
        # create candidate indices for random other wavs without current index
        indices = list(range(len(self.noisy_wavs)))
        indices.remove(index)
        rindex = random.choice(indices)
        # sample from the noisies
        rwname = os.path.join(self.data_root, self.noisy_wavs[rindex]['filename'])
        rwav, rate = sf.read(rwname)
        rwav = rwav.astype(np.float32)
        # Load current clean wav 
        uttname = self.noisy_wavs[index]['filename']
        # Santi:
        # --------------
        # TODO: Fix this shameful "path replacement" assuming there is a
        # 'noisy' -> 'clean' valid redirection from the config paths
        # Clean chunk has to be forwarded first to compute regression outputs
        # then we can load and chunk the same window on current noisy piece
        nwname = os.path.join(self.data_root, uttname)
        cwname = nwname.replace('noisy', 'clean')
        wav, rate = sf.read(cwname)
        wav = wav.astype(np.float32)
        pkg = {'raw': wav, 'raw_rand': rwav,
               'uttname': uttname, 'split': self.split}
        # Apply the set of 'target' transforms on the clean data
        if self.transform is not None:
            pkg = self.transform(pkg)

        # Load the noisy one and chunk it
        nwav, rate = sf.read(nwname)
        nwav = nwav.astype(np.float32)
        # re-direct the chunk to be cchunk (clean chunk)
        pkg['cchunk'] = pkg['chunk'].squeeze(0)
        # make the current noisy chunk
        chunk_beg = pkg['chunk_beg_i']
        chunk_end = pkg['chunk_end_i']
        chunk = nwav[chunk_beg:chunk_end]
        pkg['chunk'] = torch.FloatTensor(chunk)
        pkg['raw'] = nwav

        if self.transform is None:
            # if no transforms happened do not send a package
            return pkg['chunk'], pkg['raw_rand']
        else:
            # otherwise return the full package
            return pkg

class LibriSpeechSegTupleWavDataset(PairWavDataset):
    """ Return three wavs, one is current wav, another one is
        the continuation of a pre-chunked utterance following the name
        pattern <prefix>-<utt_id>.wav. So for example for file
        1001-134707-0001-2.wav we have to get its paired wav as
        1001-134707-0001-0.wav for instance, which is a neighbor within
        utterance level. Finally, another random and different utterance
        following the filename is returned too as random context.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rec = re.compile(r'(\d+).wav')
        # pre-cache prefixes to load from dictionary quicker
        self.neighbor_prefixes = {}
        for wav in self.wavs:
            fname = wav['filename']
            prefix = self.rec.sub('', fname)
            if prefix not in self.neighbor_prefixes:
                self.neighbor_prefixes[prefix] = []
            self.neighbor_prefixes[prefix].append(fname)
        print('Found {} prefixes in '
              'utterances'.format(len(self.neighbor_prefixes)))

    def __getitem__(self, index):
        # Load current wav or generate the zero-version
        if sample_probable(self.zero_speech_p):
            ZERO_SPEECH = True
            wav = zerospeech(int(5 * 16e3))
            cwav = wav
            uttname = 'zerospeech.wav'
        else:
            ZERO_SPEECH = False
            uttname = self.wavs[index]['filename']
            # Here we select the three wavs.
            # (1) Current wav selection
            wname = os.path.join(self.data_root, uttname)
            wav = self.retrieve_cache(wname, self.wav_cache)
            # (2) Context wav selection by utterance name pattern. If
            # no other sub-index is found, the same as current wav is returned
            prefix = self.rec.sub('', uttname)
            neighbors = self.neighbor_prefixes[prefix]
            # print('Wname: ', wname)
            # delete current file
            # print ('Uttn {}, Pref {}'.format(uttname, prefix))
            # print('Found nehg: ', neighbors)
            neighbors.remove(uttname)
            # print('Found nehg: ', neighbors)
            # pick random one if possible, otherwise it will be empty
            if len(neighbors) > 0:
                cwname = os.path.join(self.data_root, random.choice(neighbors))
                cwav = self.retrieve_cache(cwname, self.wav_cache)
            else:
                cwav = wav
        # (2) Random wav selection for out of context sample
        # create candidate indices without current index
        indices = list(range(len(self.wavs)))
        indices.remove(index)
        rindex = random.choice(indices)
        rwname = os.path.join(self.data_root, self.wavs[rindex]['filename'])
        rwav = self.retrieve_cache(rwname, self.wav_cache)
        pkg = {'raw': wav, 'raw_rand': rwav, 'raw_ctxt': cwav,
               'uttname': uttname, 'split': self.split}
        # Apply the set of 'target' transforms on the clean data
        if self.transform is not None:
            pkg = self.transform(pkg)

        pkg['cchunk'] = pkg['chunk'].squeeze(0)
        # initialize overlap label
        pkg['overlap'] = torch.zeros(pkg['chunk'].shape[-1] // pkg['dec_resolution']).float()

        if self.distortion_transforms and not ZERO_SPEECH:
            pkg = self.distortion_transforms(pkg)
        
        if self.zero_speech_transform and ZERO_SPEECH:
            pkg = self.zero_speech_transform(pkg)

        # sf.write('/tmp/ex_chunk.wav', pkg['chunk'], 16000)
        # sf.write('/tmp/ex_cchunk.wav', pkg['cchunk'], 16000)
        # raise NotImplementedError
        if self.transform is None:
            # if no transforms happened do not send a package
            return pkg['chunk'], pkg['raw_rand']
        else:
            # otherwise return the full package
            return pkg


class AmiSegTupleWavDataset(PairWavDataset):
    """ Returns 4 wavs:
    1st is IHM chunk, 2nd is continuation chunk
    3rd is a corresponding to ihm distant channel (random one from the mic array)
    4th is a random (ideally) non-related chunk 
    Note, this can also only work with only ihms (when pair_sdms=None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.zero_speech_p == 0, (
            "Zero speech mode is not supported for AMI as of now"
        )
        assert 'ihm2sdm' in kwargs, (
            "Need to provide ihm2sdm for AMI dataset"
        )
        self.ihm2sdm = None
        self.do_ihm2sdm = False
        if kwargs['ihm2sdm'] is not None:
            self.ihm2sdm = kwargs['ihm2sdm'].split(',')
            assert len(self.ihm2sdm) > 0, (
                "Expected at least one sdm channel, got {}".format(self.ihm2sdm)
            )
            self.do_ihm2sdm = True
        if self.do_ihm2sdm:
            print ('Parallel mode enabled, will pair ihm with sdms: {}'.format(self.ihm2sdm))
        else:
            print ('Single channel mode enabled, will feed only ihm data')
        # pre-cache prefixes to load from dictionary quicker
        self.neighbor_prefixes = {}
        lost_segs, lost_indices = [], []

        if self.do_ihm2sdm:
            for index, wav in enumerate(self.wavs):
                for sdm_idx in self.ihm2sdm:
                    if sdm_idx not in wav:
                        lost_segs.append(wav['filename'])
                        lost_indices.append(index)
            print ('In total {} sdm segments were missing and removed'.format(len(lost_segs)))
            for index in sorted(lost_indices, reverse=True):
                del self.wavs[index]
        
        self.rec = re.compile(r'(\d+).wav')
        for idx, wav in enumerate(self.wavs):
            fname = wav['filename']
            prefix = self.rec.sub('', fname)
            if prefix not in self.neighbor_prefixes:
                self.neighbor_prefixes[prefix] = []
            self.neighbor_prefixes[prefix].append((idx, fname))
        print('Found {} prefixes in '
              'utterances'.format(len(self.neighbor_prefixes)))

    def __getitem__(self, index):
        # Load current wav 
        # Note, this provided works with parlallel like data (i.e
        # the one where you have two or more versions of the same
        # signal, typically from multi-mic setups. As PASE relis on
        # clean and [possibly] distorted signal, the following code
        # reads both from parallel data. Notice, clean variant is only
        # used to self-supervised minions, thus is not fpropped through
        # the networks 

        # I. get and load clean (here ihm) variant as in other provider,
        # we are gonna need it anyways
        uttname = self.wavs[index]['filename']
        # Here we select the three wavs.
        # (1) Current wav selection
        wname = os.path.join(self.data_root, uttname)
        wav = self.retrieve_cache(wname, self.wav_cache)
        # (2) Context wav selection by utterance name pattern. If
        # no other sub-index is found, the same as current wav is returned
        prefix = self.rec.sub('', uttname)
        neighbors = self.neighbor_prefixes[prefix]
        # print('Wname: ', wname)
        # delete current file
        #print('Found nehg: ', neighbors)
        #print ("Uttn {}, wn {}, pref {}".format(uttname, wname, prefix))
        neighbors.remove((index, uttname))
        # print('Found nehg: ', neighbors)
        # pick random one if possible, otherwise it will be empty
        # only sample the for now candidate, we will load the wav
        # depending on whether sdm or ihm is needed
        choice = None
        if len(neighbors) > 0:
            choice = random.choice(neighbors)

        # (2) Random wav selection for out of context sample
        # create candidate indices without current index
        indices = list(range(len(self.wavs)))
        indices.remove(index)
        rindex = random.choice(indices)

        # II. depending on config, load either sdm or ihm wavs
        if self.do_ihm2sdm > 0:
            #pick random distant channel id from which to load stuff
            idx = random.choice(self.ihm2sdm)
            #print ('Utt {} idx is {}.'.format(uttname, idx))
            #print ('Index {} and cfg {}'.format(index, self.wavs[index]))
            #print ('Rindex {} and cfg {}'.format(self.wavs[rindex]))
            #if idx not in self.wavs[index]:
            #    print ('Opps {} not found in {}'.format(idx, self.wavs[index]))
            #if idx not in self.wavs[rindex]:
            #    print ('Oops {} not found in {}'.format(idx, self.wavs[rindex]))
            #load waveform sdm eqivalent for ihm
            sdm_fname = os.path.join(self.data_root, self.wavs[index][idx])
            sdm_wav = self.retrieve_cache(sdm_fname, self.wav_cache)
            #load waveform sdm random chunk
            rsdm_fname = os.path.join(self.data_root, self.wavs[rindex][idx])
            rand_sdm_wav = self.retrieve_cache(rsdm_fname, self.wav_cache)
            #load context wavform, given choice above
            if choice is not None:
                cindex, fname = choice
                cwname = os.path.join(self.data_root, self.wavs[cindex][idx])
                cwav = self.retrieve_cache(cwname, self.wav_cache)
            else:
                cwav = sdm_wav
            # Note: this one is quite dirty trick, but anyways for now
            # since we have parallel versions of data (i.e. corrputed naturally)
            # we need to extract self-supervision targets for clean, which is
            # assumed to be in chunk in all transforms. Thus, we keep it like this
            # and pass ihm wav in raw (so targets get extracted), we also pass
            # wav_sdm in raw_clean. After the transforms we swap them so sdm
            # (not ihm) chunk gets fed into the model, and ihm is preserved in cchunk
            pkg = {'raw': wav, 'raw_rand': rand_sdm_wav, 'raw_ctxt': cwav,
               'uttname': uttname, 'split': self.split, 'raw_clean':sdm_wav}
        else:
            if choice is not None:
                cindex, fname = choice
                cwname = os.path.join(self.data_root, fname)
                cwav = self.retrieve_cache(cwname, self.wav_cache)
            else:
                cwav = wav
            rwav_fname = os.path.join(self.data_root, self.wavs[rindex]['filename'])
            rwav = self.retrieve_cache(rwav_fname, self.wav_cache)
            pkg = {'raw': wav, 'raw_rand': rwav, 'raw_ctxt': cwav,
                    'uttname': uttname, 'split': self.split}

        # Apply the set of 'target' transforms on the clean data
        if self.transform is not None:
            pkg = self.transform(pkg)

        if 'cchunk' in pkg:
            chunk = pkg['cchunk']
            #print ("cchunk 1: size {}".format(chunk.size()))
            pkg['cchunk'] = pkg['chunk'].squeeze(0)
            pkg['chunk'] = chunk.squeeze(0)
            #print ("cchunk 1: size sq {}".format(pkg['cchunk'].size()))
        else:
            #print ("cchunk 2: size {}".format(pkg['chunk'].size()))
            pkg['cchunk'] = pkg['chunk'].squeeze(0)
            #print ("cchunk 2: size sq {}".format(pkg['cchunk'].size()))

        # initialize overlap label
        pkg['overlap'] = torch.zeros(len(pkg['chunk']) // pkg['dec_resolution']).float()

        if self.distortion_transforms:
            pkg = self.distortion_transforms(pkg)

        # sf.write('/tmp/ex_chunk.wav', pkg['chunk'], 16000)
        # sf.write('/tmp/ex_cchunk.wav', pkg['cchunk'], 16000)
        # raise NotImplementedError
        if self.transform is None:
            # if no transforms happened do not send a package
            return pkg['chunk'], pkg['raw_rand']
        else:
            # otherwise return the full package
            return pkg

class MetaWavConcatDataset(ConcatDataset):
    """This dataset class abstracts pool of several different datasets, 
    each having possibly a different sets of transform / distortion stacks. 
    We abstract pytorch's ConcatDataset as the code relies on several 
    dataset specific attributes (like tot_wav_dur) that are assumed to exist
    """
    def __init__(self, datasets=[]):
        super(MetaWavConcatDataset, self).__init__(datasets)
    
        for dset in self.datasets:
            assert isinstance(dset, WavDataset), (
                "{} is expected to work with WavDataset "
                "instances only.".format(__class__)
            )

    @property
    def total_wav_dur(self):
        tot_dur = 0
        for d in self.datasets:
            tot_dur += d.total_wav_dur
        return tot_dur

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
                P = self.chunker.chunk_size + 1 - len(wav)
                wav = torch.cat((wav,
                                 torch.zeros(P)),
                                dim=0)
            wav = self.chunker(wav)
            wav = wav['chunk']
        spk_id = self.utt2class[item]
        return wav, torch.LongTensor([spk_id])
