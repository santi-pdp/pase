import torch
import torch.nn.functional as F
import tqdm
import gammatone
import tempfile
from gammatone.gtgram import gtgram
import numpy as np
import subprocess
import shlex
import random
import pysptk
import os
from python_speech_features import logfbank
import librosa
import struct
import glob
import pickle
import soundfile as sf
from scipy import interpolate
from scipy import signal
from scipy.signal import decimate
from scipy.io import loadmat
import multiprocessing as mp
from scipy.signal import lfilter, resample
from scipy.interpolate import interp1d
from torchvision.transforms import Compose
from ahoproc_tools.interpolate import interpolation
from ahoproc_tools.io import *
from joblib import Parallel, delayed

try:
    import kaldi_io as kio
except ImportError:
    print ('kaldi_io is optional, but required when extracting feats with kaldi')


# Make a configurator for the distortions
def config_distortions(reverb_irfiles=None, 
                       reverb_fmt='imp',
                       reverb_data_root='.',
                       reverb_p=0.5,
                       reverb_cache=False,
                       overlap_dir=None,
                       overlap_list=None,
                       overlap_snrs=[0, 5, 10],
                       overlap_reverb=False,
                       overlap_p=0.5,
                       noises_dir=None,
                       noises_snrs=[0, 5, 10],
                       noises_p=0.5,
                       noises_cache=False,
                       speed_range=None,
                       speed_p=0.5,
                       resample_factors=[],
                       resample_p=0.5,
                       bandrop_irfiles=[],
                       bandrop_fmt='npy',
                       bandrop_data_root='.',
                       bandrop_p=0.5,
                       downsample_irfiles=[],
                       downsample_fmt='npy',
                       downsample_data_root='.',
                       downsample_p=0.5,
                       clip_factors=[], 
                       clip_p=0.5,
                       chop_factors=[],
                       #chop_factors=[(0.05, 0.025), (0.1, 0.05)], 
                       max_chops=5,
                       chop_p=0.5,
                       codec2_p=0.3,
                       codec2_kbps=1600,
                       codec2_cachedir=None,
                       codec2_cache=False,
                       report=False):
    trans = []
    probs = []
    # first of all, in case we have cached codec2 data
    if codec2_p > 0. and codec2_cachedir is not None:
        assert codec2_kbps == 1600, codec2_kbps
        trans.append(Codec2Cached(cache_dir=codec2_cachedir,
                                  cache=codec2_cache,
                                  report=report))
        probs.append(codec2_p)
    # Reverb can be shared in two different stages of the pipeline
    reverb = Reverb(reverb_irfiles, ir_fmt=reverb_fmt,
                    data_root=reverb_data_root,
                    cache=reverb_cache,
                    report=report)

    if reverb_p > 0. and reverb_irfiles is not None:
        trans.append(reverb)
        probs.append(reverb_p)

    if overlap_p > 0. and overlap_dir is not None:
        noise_trans = reverb if overlap_reverb else None
        trans.append(SimpleAdditiveShift(overlap_dir, overlap_snrs,
                                         noises_list=overlap_list,
                                         noise_transform=noise_trans,
                                         report=report))
        probs.append(overlap_p)

    if noises_p > 0. and noises_dir is not None:
        trans.append(SimpleAdditive(noises_dir, noises_snrs, 
                                    cache=noises_cache,
                                    report=report))
        probs.append(noises_p)

    if speed_p > 0. and speed_range is not None:
        # speed changer
        trans.append(SpeedChange(speed_range, report=report))
        probs.append(speed_p)

    if resample_p > 0. and len(resample_factors) > 0:
        trans.append(Resample(resample_factors, report=report))
        probs.append(resample_p)

    if clip_p > 0. and len(clip_factors) > 0:
        trans.append(Clipping(clip_factors, report=report))
        probs.append(clip_p)

    if codec2_p > 0. and codec2_cachedir is None:
        # codec2 from memory (SLOW)
        trans.append(Codec2Buffer(report=report, kbps=codec2_kbps))
        probs.append(codec2_p)

    if chop_p > 0. and len(chop_factors) > 0:
        trans.append(Chopper(max_chops=max_chops,
                             chop_factors=chop_factors,
                             report=report))
        probs.append(chop_p)
    if bandrop_p > 0. and bandrop_irfiles is not None:
        trans.append(BandDrop(bandrop_irfiles,filt_fmt=bandrop_fmt,
                              data_root=bandrop_data_root,
                              report=report))
        probs.append(bandrop_p)

    if downsample_p > 0. and len(downsample_irfiles) > 0:
        trans.append(Downsample(downsample_irfiles,filt_fmt=downsample_fmt,
                                data_root=downsample_data_root,
                                report=report))
        probs.append(downsample_p)

    if len(trans) > 0:
        return PCompose(trans, probs=probs, report=report)
    else:
        return None

def norm_and_scale(wav):
    assert isinstance(wav, torch.Tensor), type(wav)
    wav = wav / torch.max(torch.abs(wav))
    return wav * torch.rand(1)


def norm_energy(out_signal, in_signal, eps=1e-14):
    ienergy = np.dot(in_signal, in_signal)
    oenergy = np.dot(out_signal, out_signal)
    return np.sqrt(ienergy / (oenergy + eps)) * out_signal

def format_package(x):
    if not isinstance(x, dict):
        return {'raw': x}
    else:
        if 'chunk' not in x:
            x['chunk'] = x['raw']
    return x


class ToTensor(object):

    def __call__(self, pkg):
        pkg = format_package(pkg)
        for k, v in pkg.items():
            # convert everything in the package
            # into tensors
            if not isinstance(v, torch.Tensor) and not isinstance(v, str):
                pkg[k] = torch.tensor(v)
        return pkg

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ZNorm(object):

    def __init__(self, stats):
        self.stats_name = stats
        with open(stats, 'rb') as stats_f:
            self.stats = pickle.load(stats_f)

    # @profile
    def __call__(self, pkg, ignore_keys=[]):
        pkg = format_package(pkg)
        for k, st in self.stats.items():
            # assert k in pkg, '{} != {}'.format(list(pkg.keys()),
            #                                   list(self.stats.keys()))
            if k in ignore_keys:
                continue
            if k in pkg:
                mean = st['mean'].unsqueeze(1)
                std = st['std'].unsqueeze(1)
                pkg[k] = (pkg[k] - mean) / std
        return pkg

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.stats_name)


class PCompose(object):

    def __init__(self, transforms, probs=0.4, report=False):
        assert isinstance(transforms, list), type(transforms)
        self.transforms = transforms
        self.probs = probs
        self.report = report
        if isinstance(probs, list):
            assert len(transforms) == len(probs), \
                '{} != {}'.format(len(transforms),
                                  len(probs))

    #@profile
    def __call__(self, tensor):
        x = tensor
        report = {}
        for ti, transf in enumerate(self.transforms):
            if isinstance(self.probs, list):
                prob = self.probs[ti]
            else:
                prob = self.probs
            if random.random() < prob:
                x = transf(x)
                if 'report' in x:
                    # get the report
                    report = x['report']
        if self.report:
            return x, report
        else:
            return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for ti, t in enumerate(self.transforms):
            if isinstance(self.probs, list):
                prob = self.probs[ti]
            else:
                prob = self.probs
            format_string += '\n'
            format_string += '    {0}'.format(t)
            format_string += ' >> p={}'.format(prob)
        format_string += '\n)'
        return format_string


class CachedCompose(Compose):

    def __init__(self, transforms, keys, cache_path):
        super().__init__(transforms)
        self.cache_path = cache_path
        self.keys = keys
        assert len(keys) == len(transforms), '{} != {}'.format(len(keys),
                                                               len(transforms))
        print('Keys: ', keys)

    def __call__(self, x):
        if 'uttname' not in x:
            raise ValueError('Utterance name not found when '
                             'looking for cached transforms')
        if 'split' not in x:
            raise ValueError('Split name not found when '
                             'looking for cached transforms')

        znorm_ignore_flags = []
        # traverse the keys to look for cache sub-folders
        for key, t in zip(self.keys, self.transforms):
            if key == 'totensor' or key == 'chunk':
                x = t(x)
            elif key == 'znorm':
                x = t(x, znorm_ignore_flags)
            else:
                aco_dir = os.path.join(self.cache_path, x['split'], key)
                if os.path.exists(aco_dir):
                    # look for cached file by name
                    bname = os.path.splitext(os.path.basename(x['uttname']))[0]
                    acofile = os.path.join(aco_dir, bname + '.' + key)
                    if not os.path.exists(acofile):
                        acofile = None
                    else:
                        znorm_ignore_flags.append(key)
                    x = t(x, cached_file=acofile)
        return x

    def __repr__(self):
        return super().__repr__()


class SingleChunkWav(object):

    def __init__(self, chunk_size, random_scale=True,
                 pad_mode='reflect'):
        self.chunk_size = chunk_size
        self.random_scale = random_scale
        self.pad_mode = pad_mode

    def assert_format(self, x):
        # assert it is a waveform and pytorch tensor
        assert isinstance(x, torch.Tensor), type(x)
        # assert x.dim() == 1, x.size()

    #@profile
    def select_chunk(self, wav, ret_bounds=False, reuse_bounds=None):
        # select random index
        chksz = self.chunk_size
        if len(wav) <= chksz:
            # padding time
            P = chksz - len(wav)
            #if P < len(wav):
            chk = F.pad(wav.view(1, 1, -1), (0, P), 
                        mode=self.pad_mode).view(-1)
            #else:
            #    chk = F.pad(wav.view(1, 1, -1), (0, P), mode='replicate').view(-1)
            idx = 0
        elif reuse_bounds is not None:
            idx, end_i = reuse_bounds
            # padding that follows is a hack for chime, where segmenteations differ
            # between mics (by several hundred samples at most) and there may 
            # not be 1:1 correspondence between mics
            # just a fix to see if it works (its quite rara though)
            if wav.shape[0] < end_i:
                #print ("Wshape {}, beg {}, end {}".format(wav.shape[0], idx, end_i))
                if idx < wav.shape[0]:
                    chktmp = wav[idx:]
                    P = chksz - len(chktmp)
                    #print ('Len chktmp {}, P {}'.format(len(chktmp), P))
                    if P < len(chktmp):
                        chk = F.pad(chktmp.view(1, 1, -1), (0, P), mode='reflect').view(-1)
                    else:
                        chk = F.pad(chktmp.view(1, 1, -1), (0, P), mode='replicate').view(-1)
                else:
                    chk = None
            else:
                assert idx >= 0 and \
                       idx < end_i and \
                       wav.shape[0] >= end_i and \
                       chksz == end_i - idx, (
                   "Cannot reuse_bounds {} for chksz {} and wav of shape {}"\
                             .format(reuse_bounds, chksz, wav.shape)
                )
                chk = wav[idx:idx + chksz]
        else:
            # idxs = list(range(wav.size(0) - chksz))
            # idx = random.choice(idxs)
            idx = np.random.randint(0, wav.size(0) - chksz)
            chk = wav[idx:idx + chksz]
        if ret_bounds:
            return chk, idx, idx + chksz
        else:
            return chk

    def __call__(self, pkg):
        pkg = format_package(pkg)
        raw = pkg['raw']
        self.assert_format(raw)
        chunk, beg_i, end_i = self.select_chunk(raw, ret_bounds=True)
        pkg['chunk'] = chunk
        pkg['chunk_beg_i'] = beg_i
        pkg['chunk_end_i'] = end_i
        #to make it compatible with parallel multi-chan data
        #its backward compatible with single chan
        if 'raw_clean' in pkg and pkg['raw_clean'] is not None:
            raw_clean = pkg['raw_clean']
            pkg['cchunk'] = self.select_chunk(raw_clean,\
                                    reuse_bounds=(beg_i, end_i))
            if pkg['cchunk'] is None:
                #in chime5 some parallel seg does not exist, swap clean for these
                pkg['cchunk'] = pkg['chunk']
        if self.random_scale:
            pkg['chunk'] = norm_and_scale(pkg['chunk'])
            if 'cchunk' in pkg:
                pkg['cchunk'] = norm_and_scale(pkg['cchunk'])
        # specify decimated resolution to be 1 (no decimation) so far
        pkg['dec_resolution'] = 1
        return pkg

    def __repr__(self):
        return self.__class__.__name__ + \
               '({})'.format(self.chunk_size)


class MIChunkWav(SingleChunkWav):
    """ Max-Information chunker expects 3 input wavs,
        and extract 3 chunks: (chunk, chunk_ctxt,
        and chunk_rand). The first two correspond to same
        context, the third one is sampled from the second wav
    """

    def __call__(self, pkg):
        pkg = format_package(pkg)
        if 'raw_rand' not in pkg:
            raise ValueError('Need at least a pair of wavs to do '
                             'MI chunking! Just got single raw wav?')
        raw = pkg['raw']
        raw_rand = pkg['raw_rand']
        self.assert_format(raw)
        self.assert_format(raw_rand)
        chunk, beg_i, end_i = self.select_chunk(raw, ret_bounds=True)
        pkg['chunk'] = chunk
        pkg['chunk_beg_i'] = beg_i
        pkg['chunk_end_i'] = end_i
        #added for parallel like corpora with close and distant mics
        #we do not make asserts here for now if raw is 
        # exactly same as raw_clean, as this was up to segmentation
        # script
        #print ("Chunk size is {}".format(chunk.size()))
        #print ("Squeezed chunk size is {}".format(chunk.squeeze(0).size()))
        if 'raw_clean' in pkg and pkg['raw_clean'] is not None:
            raw_clean = pkg['raw_clean']
            pkg['cchunk'] = self.select_chunk(raw_clean, reuse_bounds=(beg_i, end_i))
            if pkg['cchunk'] is None:
                pkg['cchunk'] = pkg['chunk']
        if 'raw_ctxt' in pkg and pkg['raw_ctxt'] is not None:
            raw_ctxt = pkg['raw_ctxt']
        else:
            # if no additional chunk is given as raw_ctxt
            # the same as current raw context is taken
            # and a random window is selected within
            raw_ctxt = raw[:]
        pkg['chunk_ctxt'] = self.select_chunk(raw_ctxt)
        pkg['chunk_rand'] = self.select_chunk(raw_rand)
        if self.random_scale:
            pkg['chunk'] = norm_and_scale(pkg['chunk'])
            pkg['chunk_ctxt'] = norm_and_scale(pkg['chunk_ctxt'])
            pkg['chunk_rand'] = norm_and_scale(pkg['chunk_rand'])
            if 'cchunk' in pkg:
                pkg['cchunk'] = norm_and_scale(pkg['cchunk'])
        # specify decimated resolution to be 1 (no decimation) so far
        pkg['dec_resolution'] = 1
        return pkg


class LPS(object):

    def __init__(self, n_fft=2048, hop=160,
                 win=400, der_order=2,
                 name='lps',
                 device='cpu'):
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.name = name
        self.der_order=der_order
        self.device = device

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        max_frames = wav.size(0) // self.hop
        if cached_file is not None:
            # load pre-computed data
            X = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            X = X[:, beg_i:end_i]
            pkg['lps'] = X
        else:
            #print ('Chunks wav shape is {}'.format(wav.shape))
            wav = wav.to(self.device)
            X = torch.stft(wav, self.n_fft,
                           self.hop, self.win)
            X = torch.norm(X, 2, dim=2).cpu()[:, :max_frames]
            X = 10 * torch.log10(X ** 2 + 10e-20).cpu()
            if self.der_order > 0 :
                deltas=[X]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(X.numpy(),order=n))
                X=torch.from_numpy(np.concatenate(deltas))
     
            pkg[self.name] = X
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(n_fft={}, hop={}, win={}'.format(self.n_fft,
                                                   self.hop,
                                                   self.win)
        attrs += ', device={})'.format(self.device)
        return self.__class__.__name__ + attrs

class FBanks(object):

    def __init__(self, n_filters=40, n_fft=512, hop=160,
                 win=400, rate=16000, der_order=2,
                 name='fbank',
                 device='cpu'):
        self.n_fft = n_fft
        self.n_filters = n_filters
        self.rate = rate
        self.hop = hop
        self.name = name
        self.win = win
        self.der_order=der_order
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        if torch.is_tensor(wav):
            wav = wav.data.numpy().astype(np.float32)
        max_frames = wav.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            X = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            X = X[:, beg_i:end_i]
            pkg[self.name] = X
        else:
            winlen = (float(self.win) / self.rate)
            winstep = (float(self.hop) / self.rate)
            X = logfbank(wav, self.rate, winlen, winstep,
                         self.n_filters, self.n_fft).T
            expected_frames = len(wav) // self.hop

            if self.der_order > 0 :
                deltas=[X]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(X,order=n))
                X=np.concatenate(deltas)

            fbank = torch.FloatTensor(X)
            if fbank.shape[1] < expected_frames:
                P = expected_frames - fbank.shape[1]
                # pad repeating borders
                fbank = F.pad(fbank.unsqueeze(0), (0, P), mode='replicate')
                fbank = fbank.squeeze(0)
            pkg[self.name] = fbank
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(n_fft={}, n_filters={}, ' \
                'hop={}, win={}'.format(self.n_fft,
                                        self.n_filters,
                                        self.hop,
                                        self.win)
        return self.__class__.__name__ + attrs

class Gammatone(object):

    def __init__(self, f_min=500, n_channels=40, hop=160,
                 win=400,  der_order=2, rate=16000,
                 name='gtn',
                 device='cpu'):
        self.hop = hop
        self.win = win
        self.n_channels = n_channels
        self.rate = rate
        self.f_min = f_min
        self.der_order = der_order
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        if torch.is_tensor(wav):
            wav = wav.data.numpy().astype(np.float32)
        max_frames = wav.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            X = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            X = X[:, beg_i:end_i]
            pkg[self.name] = X
        else:
            windowtime = float(self.win) / self.rate
            windowhop = float(self.hop) / self.rate
            gtn = gammatone.gtgram.gtgram(wav, self.rate, 
                                          windowtime, windowhop,
                                          self.n_channels,
                                          self.f_min)
            gtn = np.log(gtn + 1e-10)
 
            if self.der_order > 0 :
                deltas=[gtn]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(gtn,order=n))
                gtn=np.concatenate(deltas)

            expected_frames = len(wav) // self.hop
            gtn = torch.FloatTensor(gtn)
            if gtn.shape[1] < expected_frames:
                P = expected_frames - gtn.shape[1]
                # pad repeating borders
                gtn = F.pad(gtn.unsqueeze(0), (0, P), mode='replicate')
                gtn = gtn.squeeze(0)
            #pkg['gtn'] = torch.FloatTensor(gtn[:, :total_frames])

            pkg[self.name] = torch.FloatTensor(gtn)
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(f_min={}, n_channels={}, ' \
                'hop={}, win={})'.format(self.f_min,
                                        self.n_channels,
                                        self.hop,
                                        self.win)
        return self.__class__.__name__ + attrs

class LPC(object):

    def __init__(self, order=25, hop=160,
                 win=320, name='lpc',
                 device='cpu'):
        self.order = order
        self.hop = hop
        self.win = win
        self.window = pysptk.hamming(win).astype(np.float32)
        self.name = name

    def frame_signal(self, signal, window):
        
        frames = []
        for beg_i in range(0, signal.shape[0], self.hop):
            frame = signal[beg_i:beg_i + self.win]
            if len(frame) < self.win:
                # pad right size with zeros
                P = self.win - len(frame)
                frame = np.concatenate((frame,
                                        np.zeros(P,)), axis=0)
            frame = frame * window
            frames.append(frame[None, :])
        frames = np.concatenate(frames, axis=0)
        return frames

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        if torch.is_tensor(wav):
            wav = wav.data.numpy().astype(np.float32)
        max_frames = wav.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            X = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            X = X[:, beg_i:end_i]
            pkg[self.name] = X
        else:
            wav = self.frame_signal(wav, self.window)
            #print('wav shape: ', wav.shape)
            lpc = pysptk.sptk.lpc(wav, order=self.order)
            #print('lpc: ', lpc.shape)
            pkg[self.name] = torch.FloatTensor(lpc)
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(order={}, hop={}, win={})'.format(self.order,
                                                    self.hop,
                                                    self.win)
        return self.__class__.__name__ + attrs

class MFCC(object):

    def __init__(self, n_fft=2048, hop=160,
                 order=13, sr=16000, win=400,
                 der_order=2, name='mfcc'):
        self.hop = hop
        # Santi: the librosa mfcc api does not always
        # accept a window argument, so we enforce n_fft
        # to be window to ensure the window len restriction
        #self.win = win
        self.n_fft = win
        self.order = order
        self.sr = 16000
        self.der_order=der_order
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            mfcc = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            mfcc = mfcc[:, beg_i:end_i]
            pkg[self.name] = mfcc
        else:
            # print(y.dtype)
            mfcc = librosa.feature.mfcc(y, sr=self.sr,
                                        n_mfcc=self.order,
                                        n_fft=self.n_fft,
                                        hop_length=self.hop,
                                        #win_length=self.win,
                                        )[:, :max_frames]
            if self.der_order > 0 :
                deltas=[mfcc]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(mfcc,order=n))
                mfcc=np.concatenate(deltas)
    
            pkg[self.name] = torch.tensor(mfcc.astype(np.float32))
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(order={}, sr={})'.format(self.order,
                                           self.sr)
        return self.__class__.__name__ + attrs

class MFCC_librosa(object):

    def __init__(self, n_fft=2048, hop=160,
                 order=13, sr=16000, win=400,der_order=2,n_mels=40,
                 htk=True, name='mfcc_librosa'):
        self.hop = hop
        # Santi: the librosa mfcc api does not always
        # accept a window argument, so we enforce n_fft
        # to be window to ensure the window len restriction
        #self.win = win
        self.n_fft = win
        self.order = order
        self.sr = 16000
        self.der_order=der_order
        self.n_mels=n_mels
        self.htk=True
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            mfcc = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            mfcc = mfcc[:, beg_i:end_i]
            pkg[self.name] = mfcc
        else:
            # print(y.dtype)
            mfcc = librosa.feature.mfcc(y, sr=self.sr,
                                        n_mfcc=self.order,
                                        n_fft=self.n_fft,
                                        hop_length=self.hop,
                                        #win_length=self.win,
					n_mels=self.n_mels,
                                        htk=self.htk,
                                        )[:, :max_frames]
            if self.der_order > 0 :
                deltas=[mfcc]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(mfcc,order=n))
                mfcc=np.concatenate(deltas)

            pkg[self.name] = torch.tensor(mfcc.astype(np.float32))
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(order={}, sr={})'.format(self.order,
                                           self.sr)
        return self.__class__.__name__ + attrs

class KaldiFeats(object):
    def __init__(self, kaldi_root, hop=160, win=400, sr=16000):

        if kaldi_root is None and 'KALDI_ROOT' in os.environ:
            kaldi_root = os.environ['KALDI_ROOT']

        assert kaldi_root is not None, (
            "Set KALDI_ROOT (either pass via cmd line, or set env variable)"
        )

        self.kaldi_root = kaldi_root
        self.hop = hop
        self.win = win
        self.sr = sr

        self.frame_shift = int(1000./self.sr * self.hop) #in ms
        self.frame_length = int(1000./self.sr * self.win) #in ms

    def __execute_command__(self, datain, cmd):
        #try:
        fin, fout = kio.open_or_fd(cmd, 'wb')
        kio.write_wav(fin, datain, self.sr, key='utt')
        fin.close() #so its clear nothing new arrives
        feats_ark = kio.read_mat_ark(fout)
        for _, feats in feats_ark:
            fout.close()
            return feats.T #there is only one to read
        #except Exception as e:
        #    print (e)
        #    return None

    def __repr__(self):
        return self.__class__.__name__

class KaldiMFCC(KaldiFeats):
    def __init__(self, kaldi_root, hop=160, win=400, sr=16000,
                    num_mel_bins=40, num_ceps=13, der_order=2,
                    name='kaldimfcc'):

        super(KaldiMFCC, self).__init__(kaldi_root=kaldi_root, 
                                        hop=hop, win=win, sr=sr)

        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.der_order=der_order

        cmd = "ark:| {}/src/featbin/compute-mfcc-feats --print-args=false "\
               "--use-energy=false --snip-edges=false --num-ceps={} "\
               "--frame-length={} --frame-shift={} "\
               "--num-mel-bins={} --sample-frequency={} ark:- ark:- |"\
               " {}/src/featbin/add-deltas --print-args=false "\
               "--delta-order={} ark:- ark:- |"

        self.cmd = cmd.format(self.kaldi_root, self.num_ceps,
                              self.frame_length, self.frame_shift,
                              self.num_mel_bins, self.sr, self.kaldi_root,
                              self.der_order)
        self.name = name

    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            mfcc = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            mfcc = mfcc[:, beg_i:end_i]
            pkg[self.name] = mfcc
        else:
            # print(y.dtype)
            mfccs = self.__execute_command__(y, self.cmd)
            assert mfccs is not None, (
                "Mfccs extraction failed"
            )
            pkg[self.name] = torch.tensor(mfccs[:,:max_frames].astype(np.float32))

        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = "(bins={}, ceps={}, sr={})"\
                  .format(self.num_mel_bins, self.num_ceps, self.sr)
        return self.__class__.__name__ + attrs

class KaldiPLP(KaldiFeats):
    def __init__(self, kaldi_root, hop=160, win=400, sr=16000,
                 num_mel_bins=20, num_ceps=20, lpc_order=20,
                 name='kaldiplp'):

        super(KaldiPLP, self).__init__(kaldi_root=kaldi_root, 
                                        hop=hop, win=win, sr=sr)

        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.lpc_order = lpc_order

        cmd = "ark:| {}/src/featbin/compute-plp-feats "\
               "--print-args=false --snip-edges=false --use-energy=false "\
               "--num-ceps={} --lpc-order={} "\
               "--frame-length={} --frame-shift={} "\
               "--num-mel-bins={} --sample-frequency={} "\
               "ark:- ark:- |"

        self.cmd = cmd.format(self.kaldi_root, self.num_ceps, self.lpc_order, 
                              self.frame_length, self.frame_shift, 
                              self.num_mel_bins, self.sr)
        self.name = name

    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            plp = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            plp = plp[:, beg_i:end_i]
            pkg[self.name] = plp
        else:
            # print(y.dtype)
            feats = self.__execute_command__(y, self.cmd)
            pkg[self.name] = torch.tensor(feats[:,:max_frames].astype(np.float32))
        
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = "(bins={}, ceps={}, sr={}, lpc={})"\
                  .format(self.num_mel_bins, self.num_ceps, self.sr, self.lpc_order)
        return self.__class__.__name__ + attrs

class Prosody(object):

    def __init__(self, hop=160, win=320, f0_min=60, f0_max=300,der_order=2,
                 sr=16000, name='prosody'):
        self.hop = hop
        self.win = win
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sr = sr
        self.der_order = der_order
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy()
        max_frames = wav.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            proso = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            proso = proso[:, beg_i:end_i]
            pkg[self.name] = proso
        else:
            # first compute logF0 and voiced/unvoiced flag
            # f0 = pysptk.rapt(wav.astype(np.float32),
            #                 fs=self.sr, hopsize=self.hop,
            #                 min=self.f0_min, max=self.f0_max,
            #                 otype='f0')
            f0 = pysptk.swipe(wav.astype(np.float64),
                              fs=self.sr, hopsize=self.hop,
                              min=self.f0_min,
                              max=self.f0_max,
                              otype='f0')
            # sound = pm.Sound(wav.astype(np.float32), self.sr)
            # f0 = sound.to_pitch(self.hop / 16000).selected_array['frequency']
            if len(f0) < max_frames:
                pad = max_frames - len(f0)
                f0 = np.concatenate((f0, f0[-pad:]), axis=0)
            lf0 = np.log(f0 + 1e-10)
            lf0, uv = interpolation(lf0, -1)
            lf0 = torch.tensor(lf0.astype(np.float32)).unsqueeze(0)[:, :max_frames]
            uv = torch.tensor(uv.astype(np.float32)).unsqueeze(0)[:, :max_frames]
            if torch.sum(uv) == 0:
                # if frame is completely unvoiced, make lf0 min val
                lf0 = torch.ones(uv.size()) * np.log(self.f0_min)
            # assert lf0.min() > 0, lf0.data.numpy()
            # secondly obtain zcr
            zcr = librosa.feature.zero_crossing_rate(y=wav,
                                                     frame_length=self.win,
                                                     hop_length=self.hop)
            zcr = torch.tensor(zcr.astype(np.float32))
            zcr = zcr[:, :max_frames]
            # finally obtain energy
            egy = librosa.feature.rmse(y=wav, frame_length=self.win,
                                       hop_length=self.hop,
                                       pad_mode='constant')
            egy = torch.tensor(egy.astype(np.float32))
            egy = egy[:, :max_frames]
            proso = torch.cat((lf0, uv, egy, zcr), dim=0)
  
            if self.der_order > 0 :
                deltas=[proso]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(proso.numpy(),order=n))
                proso=torch.from_numpy(np.concatenate(deltas))

            pkg[self.name] = proso
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(hop={}, win={}, f0_min={}, f0_max={}'.format(self.hop,
                                                               self.win,
                                                               self.f0_min,
                                                               self.f0_max)
        attrs += ', sr={})'.format(self.sr)
        return self.__class__.__name__ + attrs

class Reverb(object):

    def __init__(self, ir_files, report=False, ir_fmt='mat',
                 max_reverb_len=24000,
                 cache=False,
                 data_root='.'):
        if len(ir_files) == 0:
            # list the directory
            ir_files = [os.path.basename(f) for f in glob.glob(os.path.join(data_root,
                                              '*.{}'.format(ir_fmt)))]
            print('Found {} *.{} ir_files in {}'.format(len(ir_files),
                                                        ir_fmt,
                                                        data_root))
        self.ir_files = ir_files
        assert isinstance(ir_files, list), type(ir_files)
        assert len(ir_files) > 0, len(ir_files)
        self.ir_idxs = list(range(len(ir_files)))
        # self.IR, self.p_max = self.load_IR(ir_file, ir_fmt)
        self.ir_fmt = ir_fmt
        self.report = report
        self.data_root = data_root
        self.max_reverb_len = max_reverb_len
        if cache:
            self.cache = {}
            for ir_file in self.ir_files:
                self.load_IR(ir_file, ir_fmt)

    def load_IR(self, ir_file, ir_fmt):
        ir_file = os.path.join(self.data_root, ir_file)
        # print('loading ir_file: ', ir_file)
        if hasattr(self, 'cache') and ir_file in self.cache:
            return self.cache[ir_file]
        else:
            if ir_fmt == 'mat':
                IR = loadmat(ir_file, squeeze_me=True, struct_as_record=False)
                IR = IR['risp_imp']
            elif ir_fmt == 'imp' or ir_fmt == 'txt':
                IR = np.loadtxt(ir_file)
            elif ir_fmt == 'npy':
                IR = np.load(ir_file)
            elif ir_fmt == 'wav':
                IR, _ = sf.read(ir_file)
            else:
                raise TypeError('Unrecognized IR format: ', ir_fmt)
            IR = IR[:self.max_reverb_len]
            if np.max(IR)>0:
                IR = IR / np.abs(np.max(IR))
            p_max = np.argmax(np.abs(IR))
            if hasattr(self, 'cache'):
                self.cache[ir_file] = (IR, p_max)
            return IR, p_max

    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e

    def sample_IR(self):
        if len(self.ir_files) == 0:
            return self.ir_files[0]
        else:
            idx = random.choice(self.ir_idxs)
            return self.ir_files[idx]

    ##@profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        # sample an ir_file
        ir_file = self.sample_IR()
        IR, p_max = self.load_IR(ir_file, self.ir_fmt)
        IR = IR.astype(np.float32)
        wav = wav.data.numpy().reshape(-1)
        Ex = np.dot(wav, wav)
        wav = wav.astype(np.float32).reshape(-1)
        # wav = wav / np.max(np.abs(wav))
        # rev = signal.fftconvolve(wav, IR, mode='full')
        rev = signal.convolve(wav, IR, mode='full').reshape(-1)
        Er = np.dot(rev, rev)
        # rev = rev / np.max(np.abs(rev))
        # IR delay compensation
        rev = self.shift(rev, -p_max)
        if Er > 0:
            Eratio = np.sqrt(Ex / Er) 
        else:
            Eratio = 1.0
            #rev = rev / np.max(np.abs(rev))

        # Trim rev signal to match clean length
        rev = rev[:wav.shape[0]]
        rev = Eratio * rev
        rev = torch.FloatTensor(rev)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['ir_file'] = ir_file
        pkg['chunk'] = rev
        return pkg

    def __repr__(self):
        if len(self.ir_files) > 3:
            attrs = '(ir_files={} ...)'.format(self.ir_files[:3])
        else:
            attrs = '(ir_files={})'.format(self.ir_files)
        return self.__class__.__name__ + attrs


class Downsample(object):

    def __init__(self, filt_files, report=False, filt_fmt='npy',
                 data_root='.'):
        self.filt_files = filt_files
        assert isinstance(filt_files, list), type(filt_files)
        assert len(filt_files) > 0, len(filt_files)
        self.filt_idxs = list(range(len(filt_files)))
        self.filt_fmt = filt_fmt
        self.report = report
        self.data_root = data_root

    def load_filter(self, filt_file, filt_fmt):

        filt_file = os.path.join(self.data_root, filt_file)

        if filt_fmt == 'mat':
            filt_coeff = loadmat(filt_file, squeeze_me=True, struct_as_record=False)
            filt_coeff = filt_coeff['filt_coeff']

        elif filt_fmt == 'imp' or filt_fmt == 'txt':
            filt_coeff = np.loadtxt(filt_file)
        elif filt_fmt == 'npy':
            filt_coeff = np.load(filt_file)
        else:
            raise TypeError('Unrecognized filter format: ', filt_fmt)

        filt_coeff = filt_coeff / np.abs(np.max(filt_coeff))

        return filt_coeff

    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e

    def sample_filt(self):
        if len(self.filt_files) == 0:
            return self.filt_files[0]
        else:
            idx = random.choice(self.filt_idxs)
            return self.filt_files[idx]

    ##@profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        # sample a filter
        filt_file = self.sample_filt()
        filt_coeff = self.load_filter(filt_file, self.filt_fmt)
        filt_coeff = filt_coeff.astype(np.float32)
        wav = wav.data.numpy().reshape(-1)
        Ex = np.dot(wav, wav)
        wav = wav.astype(np.float32).reshape(-1)

        sig_filt = signal.convolve(wav, filt_coeff, mode='full').reshape(-1)

        sig_filt = self.shift(sig_filt, -round(filt_coeff.shape[0] / 2))

        sig_filt = sig_filt[:wav.shape[0]]

        # sig_filt=sig_filt/np.max(np.abs(sig_filt))

        Efilt = np.dot(sig_filt, sig_filt)
        # Ex = np.dot(wav, wav)

        if Efilt > 0:
            Eratio = np.sqrt(Ex / Efilt)
        else:
            Eratio = 1.0
            sig_filt = wav

        sig_filt = Eratio * sig_filt
        sig_filt = torch.FloatTensor(sig_filt)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['filt_file'] = filt_file
        pkg['chunk'] = sig_filt
        return pkg

    def __repr__(self):
        if len(self.filt_files) > 3:
            attrs = '(filt_files={} ...)'.format(self.filt_files[:3])
        else:
            attrs = '(filt_files={})'.format(self.filt_files)
        return self.__class__.__name__ + attrs


class BandDrop(object):

    def __init__(self, filt_files, report=False, filt_fmt='npy',
                 data_root='.'):
        if len(filt_files) == 0:
            # list the directory
            filt_files = [os.path.basename(f) for f in glob.glob(os.path.join(data_root,
                                              '*.{}'.format(filt_fmt)))]
            print('Found {} *.{} filt_files in {}'.format(len(filt_files),
                                                          filt_fmt,
                                                          data_root))
        self.filt_files = filt_files
        assert isinstance(filt_files, list), type(filt_files)
        assert len(filt_files) > 0, len(filt_files)
        self.filt_idxs = list(range(len(filt_files)))
        self.filt_fmt = filt_fmt
        self.report = report
        self.data_root = data_root

    def load_filter(self, filt_file, filt_fmt):

        filt_file = os.path.join(self.data_root, filt_file)

        if filt_fmt == 'mat':
            filt_coeff = loadmat(filt_file, squeeze_me=True, struct_as_record=False)
            filt_coeff = filt_coeff['filt_coeff']

        elif filt_fmt == 'imp' or filt_fmt == 'txt':
            filt_coeff = np.loadtxt(filt_file)
        elif filt_fmt == 'npy':
            filt_coeff = np.load(filt_file)
        else:
            raise TypeError('Unrecognized filter format: ', filt_fmt)

        filt_coeff = filt_coeff / np.abs(np.max(filt_coeff))

        return filt_coeff

    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e

    def sample_filt(self):
        if len(self.filt_files) == 0:
            return self.filt_files[0]
        else:
            idx = random.choice(self.filt_idxs)
            return self.filt_files[idx]

    ##@profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        # sample a filter
        filt_file = self.sample_filt()
        filt_coeff = self.load_filter(filt_file, self.filt_fmt)
        filt_coeff = filt_coeff.astype(np.float32)
        wav = wav.data.numpy().reshape(-1)
        Ex = np.dot(wav, wav)
        wav = wav.astype(np.float32).reshape(-1)

        sig_filt = signal.convolve(wav, filt_coeff, mode='full').reshape(-1)

        sig_filt = self.shift(sig_filt, -round(filt_coeff.shape[0] / 2))

        sig_filt = sig_filt[:wav.shape[0]]

        # sig_filt=sig_filt/np.max(np.abs(sig_filt))

        Efilt = np.dot(sig_filt, sig_filt)
        # Ex = np.dot(wav, wav)
        if Efilt > 0:
            Eratio = np.sqrt(Ex / Efilt)
        else:
            Eratio = 1.0
            sig_filt = wav

        sig_filt = Eratio * sig_filt
        sig_filt = torch.FloatTensor(sig_filt)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['filt_file'] = filt_file
        pkg['chunk'] = sig_filt
        return pkg

    def __repr__(self):
        if len(self.filt_files) > 3:
            attrs = '(filt_files={} ...)'.format(self.filt_files[:3])
        else:
            attrs = '(filt_files={})'.format(self.filt_files)
        return self.__class__.__name__ + attrs


class Scale(object):
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".
    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth
    """

    def __init__(self, factor=2 ** 31):
        self.factor = factor

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)
        Returns:
            Tensor: Scaled by the scale factor. (default between -1.0 and 1.0)
        """
        if isinstance(tensor, (torch.LongTensor, torch.IntTensor)):
            tensor = tensor.float()

        return tensor / self.factor


class SimpleChopper(object):
    """ Do not use VAD to specify speech regions, just
        cut randomly some number of regions randomly
    """

    def __init__(self, chop_factors=[(0.05, 0.025), (0.1, 0.05)],
                 max_chops=5, report=False):
        self.chop_factors = chop_factors
        self.max_chops = max_chops
        self.report = report

    def chop_wav(self, wav):
        # TODO: finish this
        raise NotImplementedError('Need to be finished')
        chop_factors = self.chop_factors
        # get num of chops to make
        chops = np.random.randint(1, self.max_chops + 1)
        # build random indexes to randomly pick regions, not ordered
        if chops == 1:
            chop_idxs = [0]
        else:
            chop_idxs = np.random.choice(list(range(chops)), chops,
                                         replace=False)
        chopped_wav = np.copy(wav)
        return None

    def __call__(self, pkg, srate=16000):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        # unorm to 16-bit scale for VAD in chopper
        wav = wav.data.numpy().astype(np.float32)
        # get speech regions for proper chopping
        chopped = self.chop_wav(wav)
        chopped = self.normalizer(torch.FloatTensor(chopped))
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['speech_regions'] = speech_regions
        pkg['chunk'] = chopped
        return pkg

    def __repr__(self):
        attrs = '(chop_factors={}, max_chops={})'.format(
            self.chop_factors,
            self.max_chops
        )


class Chopper(object):
    def __init__(self, chop_factors=[(0.05, 0.025), (0.1, 0.05)],
                 max_chops=2, force_regions=False, report=False):
        # chop factors in seconds (mean, std) per possible chop
        import webrtcvad
        self.chop_factors = chop_factors
        self.max_chops = max_chops
        self.force_regions = force_regions
        # create VAD to get speech chunks
        self.vad = webrtcvad.Vad(2)
        # make scalers to norm/denorm
        self.denormalizer = Scale(1. / ((2 ** 15) - 1))
        self.normalizer = Scale((2 ** 15) - 1)
        self.report = report

    # @profile
    def vad_wav(self, wav, srate):
        """ Detect the voice activity in the 16-bit mono PCM wav and return
            a list of tuples: (speech_region_i_beg_sample, center_sample,
            region_duration)
        """
        if srate != 16000:
            raise ValueError('Sample rate must be 16kHz')
        window_size = 160  # samples
        regions = []
        curr_region_counter = 0
        init = None
        vad = self.vad
        if self.force_regions:
            # Divide the signal into even regions depending on number of chops
            # to put
            nregions = wav.shape[0] // self.max_chops
            reg_len = wav.shape[0] // nregions
            for beg_i in range(0, wav.shape[0], reg_len):
                end_sample = beg_i + reg_len
                center_sample = beg_i + (end_sample - beg_i) / 2
                regions.append((beg_i, center_sample,
                                reg_len))
            return regions
        else:
            # Use the VAD to determine actual speech regions
            for beg_i in range(0, wav.shape[0], window_size):
                frame = wav[beg_i:beg_i + window_size]
                if frame.shape[0] >= window_size and \
                        vad.is_speech(struct.pack('{}i'.format(window_size),
                                                  *frame), srate):
                    curr_region_counter += 1
                    if init is None:
                        init = beg_i
                else:
                    # end of speech region (or never began yet)
                    if init is not None:
                        # close the region
                        end_sample = init + (curr_region_counter * window_size)
                        center_sample = init + (end_sample - init) / 2
                        regions.append((init, center_sample,
                                        curr_region_counter * window_size))
                    init = None
                    curr_region_counter = 0
            return regions

    # @profile
    def chop_wav(self, wav, srate, speech_regions):
        if len(speech_regions) == 0:
            # print('Skipping no speech regions')
            return wav, []
        chop_factors = self.chop_factors
        # get num of chops to make
        num_chops = list(range(1, self.max_chops + 1))
        chops = np.asscalar(np.random.choice(num_chops, 1))
        # trim it to available regions
        chops = min(chops, len(speech_regions))
        #print('Making {} chops'.format(chops))
        # build random indexes to randomly pick regions, not ordered
        if chops == 1:
            chop_idxs = [0]
        else:
            chop_idxs = np.random.choice(list(range(chops)), chops,
                                         replace=False)
        chopped_wav = np.copy(wav)
        chops_log = []
        # make a chop per chosen region
        for chop_i in chop_idxs:
            region = speech_regions[chop_i]
            # decompose the region
            reg_beg, reg_center, reg_dur = region
            # pick random chop_factor
            chop_factor_idx = np.random.choice(range(len(chop_factors)), 1)[0]
            chop_factor = chop_factors[chop_factor_idx]
            # compute duration from: std * N(0, 1) + mean
            mean, std = chop_factor
            chop_dur = mean + np.random.randn(1) * std
            # convert dur to samples
            chop_s_dur = int(chop_dur * srate)
            chop_beg = max(int(reg_center - (chop_s_dur / 2)), reg_beg)
            chop_end = min(int(reg_center + (chop_s_dur / 2)), reg_beg +
                           reg_dur)
            # print('chop_beg: ', chop_beg)
            # print('chop_end: ', chop_end)
            # chop the selected region with computed dur
            chopped_wav[chop_beg:chop_end] = 0
            chops_log.append(float(chop_dur))
        return chopped_wav, chops_log

    #@profile
    def __call__(self, pkg, srate=16000):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        # unorm to 16-bit scale for VAD in chopper
        wav = self.denormalizer(wav)
        wav = wav.data.numpy()
        wav = wav.astype(np.int16)
        if wav.ndim > 1:
            wav = wav.reshape((-1,))
        # get speech regions for proper chopping
        speech_regions = self.vad_wav(wav, srate)
        chopped, chops = self.chop_wav(wav, srate,
                                       speech_regions)
        chopped = chopped.astype(np.float32)
        chopped = self.normalizer(torch.from_numpy(chopped))
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['chops'] = chops
        pkg['chunk'] = chopped
        return pkg

    def __repr__(self):
        attrs = '(chop_factors={}, max_chops={})'.format(
            self.chop_factors,
            self.max_chops
        )
        return self.__class__.__name__ + attrs


class Clipping(object):

    def __init__(self, clip_factors=[0.3, 0.4, 0.5],
                 report=False):
        self.clip_factors = clip_factors
        self.report = report

    #@profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy().astype(np.float32)
        # cf = np.random.choice(self.clip_factors, 1)
        cf = random.choice(self.clip_factors)
        clip = np.maximum(wav, cf * np.min(wav))
        clip = np.minimum(clip, cf * np.max(wav))
        clipT = torch.FloatTensor(clip)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['clip_factor'] = cf
        pkg['chunk'] = clipT
        return pkg

    def __repr__(self):
        attrs = '(clip_factors={})'.format(
            self.clip_factors
        )
        return self.__class__.__name__ + attrs


class Resample(object):

    def __init__(self, factors=[4], report=False):
        self.factors = factors
        self.report = report

    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy()
        factor = random.choice(self.factors)
        x_lr = decimate(wav, factor).copy()
        x_lr = torch.FloatTensor(x_lr)
        x_ = F.interpolate(x_lr.view(1, 1, -1),
                           scale_factor=factor,
                           align_corners=True,
                           mode='linear').view(-1)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['resample_factor'] = factor
        pkg['chunk'] = x_
        return pkg

    def __repr__(self):
        attrs = '(factor={})'.format(
            self.factors
        )
        return self.__class__.__name__ + attrs


class SimpleAdditive(object):

    def __init__(self, noises_dir, snr_levels=[0, 5, 10],
                 cache=False,
                 report=False):
        self.noises_dir = noises_dir
        self.snr_levels = snr_levels
        self.report = report
        # read noises in dir
        if isinstance(noises_dir, list):
            self.noises = []
            for ndir in noises_dir:
                self.noises += glob.glob(os.path.join(ndir, '*.wav'))
        else:
            self.noises = glob.glob(os.path.join(noises_dir, '*.wav'))
        self.nidxs = list(range(len(self.noises)))
        if len(self.noises) == 0:
            raise ValueError('[!] No noises found in {}'.format(noises_dir))
        else:
            print('[*] Found {} noise files'.format(len(self.noises)))
        self.eps = 1e-22
        if cache:
            self.cache = {}
            for noise in self.noises:
                self.load_noise(noise)

    def sample_noise(self):
        if len(self.noises) == 1:
            return self.noises[0]
        else:
            idx = np.random.randint(0, len(self.noises))
            # idx = random.choice(self.nidxs)
            return self.noises[idx]

    def load_noise(self, filename):
        if hasattr(self, 'cache') and filename in self.cache:
            return self.cache[filename]
        else:
            nwav, rate = sf.read(filename)
            if hasattr(self, 'cache'):
                self.cache[filename] = nwav
        return nwav

    def compute_SNR_K(self, signal, noise, snr):
        Ex = np.dot(signal, signal)
        En = np.dot(noise, noise)
        if En > 0:
            K = np.sqrt(Ex / ((10 ** (snr / 10.)) * En))
        else:
            K = 1.0
        return K, Ex, En

    def norm_energy(self, osignal, ienergy, eps=1e-14):
        oenergy = np.dot(osignal, osignal)
        return np.sqrt(ienergy / (oenergy + eps)) * osignal

    #@profile
    def __call__(self, pkg):
        """ Add noise to clean wav """
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy().reshape(-1)
        if 'chunk_beg_i' in pkg:
            beg_i = pkg['chunk_beg_i']
            end_i = pkg['chunk_end_i']
        else:
            beg_i = 0
            end_i = wav.shape[0]
        sel_noise = self.load_noise(self.sample_noise())
        if len(sel_noise) < len(wav):
            # pad noise
            P = len(wav) - len(sel_noise)
            sel_noise = F.pad(torch.tensor(sel_noise).view(1, 1, -1),
                              (0, P),
                              ).view(-1).data.numpy()
                              #mode='reflect').view(-1).data.numpy()
        T = end_i - beg_i
        # TODO: not pre-loading noises from files?
        if len(sel_noise) > T:
            n_beg_i = np.random.randint(0, len(sel_noise) - T)
        else:
            n_beg_i = 0
        noise = sel_noise[n_beg_i:n_beg_i + T].astype(np.float32)
        # randomly sample the SNR level
        snr = random.choice(self.snr_levels)
        K, Ex, En = self.compute_SNR_K(wav, noise, snr)
        scaled_noise = K * noise
        if En > 0:
            noisy = wav + scaled_noise
            noisy = self.norm_energy(noisy, Ex)
        else:
            noisy = wav

        x_ = torch.FloatTensor(noisy)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['snr'] = snr
        pkg['chunk'] = x_
        return pkg

    def __repr__(self):
        attrs = '(noises_dir={})'.format(
            self.noises_dir
        )
        return self.__class__.__name__ + attrs


class SimpleAdditiveShift(SimpleAdditive):

    def __init__(self, noises_dir, snr_levels=[5, 10],
                 noise_transform=None,
                 noises_list=None,
                 report=False):
        if noises_list is None:
            super().__init__(noises_dir, snr_levels, report)
        else:
            if isinstance(noises_dir, list):
                assert len(noises_dir) == 1, len(noises_dir)
                noises_dir = noises_dir[0]
            with open(noises_list, 'r') as nf:
                self.noises = []
                for nel in nf:
                    nel = nel.rstrip()
                    self.noises.append(os.path.join(noises_dir, nel))
        self.noises_dir = noises_dir
        self.noises_list = noises_list
        self.snr_levels = snr_levels
        self.report = report
        self.nidxs = list(range(len(self.noises)))
        if len(self.noises) == 0:
            raise ValueError('[!] No noises found in {}'.format(noises_dir))
        else:
            print('[*] Found {} noise files'.format(len(self.noises)))
        # additional out_transform to include potential distortions
        self.noise_transform = noise_transform

    #@profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy().reshape(-1)
        # compute shifts of signal
        shift = np.random.randint(0, int(0.75 * len(wav)))
        sel_noise = self.load_noise(self.sample_noise())
        T = len(wav) - shift
        if len(sel_noise) < T:
            # pad noise
            P = T - len(sel_noise)
            sel_noise = F.pad(torch.tensor(sel_noise).view(1, 1, -1),
                              (0, P),
                              mode='constant').view(-1).data.numpy()
            n_beg_i = 0
        elif len(sel_noise) > T:
            n_beg_i = np.random.randint(0, len(sel_noise) - T)
        else:
            n_beg_i = 0
        noise = sel_noise[n_beg_i:n_beg_i + T].astype(np.float32)
        if self.noise_transform is not None:
            noise = self.noise_transform({'chunk': torch.FloatTensor(noise)})['chunk']
            noise = noise.data.numpy()
        pad_len = len(wav) - len(noise)
        if 'overlap' in pkg:
            # anotate a mask of overlapped samples
            dec_res = pkg['dec_resolution'] 
            dec_len = len(wav) // dec_res
            #assert dec_len == len(pkg['overlap']), dec_len
            pkg['overlap'] = torch.cat((torch.zeros(pad_len),
                                       torch.ones(len(noise))),
                                       dim=0).float()
            if dec_res > 1:
                to_dec = pkg['overlap'].view(-1, dec_res)
                pkg['overlap'] = torch.mean(to_dec, dim=1)

        # apply padding to equal length now
        noise = F.pad(torch.tensor(noise).view(1, 1, -1),
                      (pad_len, 0),
                      mode='constant').view(-1).data.numpy()
        # randomly sample the SNR level
        snr = random.choice(self.snr_levels)
        K, Ex, En = self.compute_SNR_K(wav, noise, snr)
        scaled_noise = K * noise
        noisy = wav + scaled_noise
        noisy = self.norm_energy(noisy, Ex)
        x_ = torch.FloatTensor(noisy)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['snr'] = snr
        pkg['chunk'] = x_
        return pkg

    def __repr__(self):
        if self.noise_transform is None:
            attrs = '(noises_dir={})'.format(
                self.noises_dir
            )
        else:
            attrs = '(noises_dir={}, noises_list={}, ' \
                    'noise_transform={})'.format(
                self.noises_dir,
                self.noises_list,
                self.noise_transform.__repr__()
            )
        return self.__class__.__name__ + attrs


class Additive(object):

    def __init__(self, noises_dir, snr_levels=[0, 5, 10], do_IRS=False,
                 prob=1):
        self.prob = prob
        self.noises_dir = noises_dir
        self.snr_levels = snr_levels
        self.do_IRS = do_IRS
        # read noises in dir
        noises = glob.glob(os.path.join(noises_dir, '*.wav'))
        if len(noises) == 0:
            raise ValueError('[!] No noises found in {}'.format(noises_dir))
        else:
            print('[*] Found {} noise files'.format(len(noises)))
            self.noises = []
            for n_i, npath in enumerate(noises, start=1):
                # nwav = wavfile.read(npath)[1]
                nwav = librosa.load(npath, sr=None)[0]
                self.noises.append({'file': npath,
                                    'data': nwav.astype(np.float32)})
                log_noise_load = 'Loaded noise {:3d}/{:3d}: ' \
                                 '{}'.format(n_i, len(noises),
                                             npath)
                print(log_noise_load)
        self.eps = 1e-22

    def __call__(self, wav, srate=16000, nbits=16):
        """ Add noise to clean wav """
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()
        noise_idx = np.random.choice(list(range(len(self.noises))), 1)
        sel_noise = self.noises[np.asscalar(noise_idx)]
        noise = sel_noise['data']
        snr = np.random.choice(self.snr_levels, 1)
        # print('Applying SNR: {} dB'.format(snr[0]))
        if wav.ndim > 1:
            wav = wav.reshape((-1,))
        noisy, noise_bound = self.addnoise_asl(wav, noise, srate,
                                               nbits, snr,
                                               do_IRS=self.do_IRS)
        # normalize to avoid clipping
        if np.max(noisy) >= 1 or np.min(noisy) < -1:
            small = 0.1
            while np.max(noisy) >= 1 or np.min(noisy) < -1:
                noisy = noisy / (1. + small)
                small = small + 0.1
        return torch.FloatTensor(noisy.astype(np.float32))

    def addnoise_asl(self, clean, noise, srate, nbits, snr, do_IRS=False):
        if do_IRS:
            # Apply IRS filter simulating telephone
            # handset BW [300, 3200] Hz
            clean = self.apply_IRS(clean, srate, nbits)
        Px, asl, c0 = self.asl_P56(clean, srate, nbits)
        # Px is active speech level ms energy
        # asl is active factor
        # c0 is active speech level threshold
        x = clean
        x_len = x.shape[0]

        noise_len = noise.shape[0]
        if noise_len <= x_len:
            print('Noise length: ', noise_len)
            print('Speech length: ', x_len)
            raise ValueError('Noise length has to be greater than speech '
                             'length!')
        rand_start_limit = int(noise_len - x_len + 1)
        rand_start = int(np.round((rand_start_limit - 1) * np.random.rand(1) \
                                  + 1))
        noise_segment = noise[rand_start:rand_start + x_len]
        noise_bounds = (rand_start, rand_start + x_len)

        if do_IRS:
            noise_segment = self.apply_IRS(noise_segment, srate, nbits)

        Pn = np.dot(noise_segment.T, noise_segment) / x_len

        # we need to scale the noise segment samples to obtain the
        # desired SNR = 10 * log10( Px / ((sf ** 2) * Pn))
        sf = np.sqrt(Px / Pn / (10 ** (snr / 10)))
        noise_segment = noise_segment * sf

        noisy = x + noise_segment

        return noisy, noise_bounds

    def apply_IRS(self, data, srate, nbits):
        """ Apply telephone handset BW [300, 3200] Hz """
        raise NotImplementedError('Under construction!')
        from pyfftw.interfaces import scipy_fftpack as fftw
        n = data.shape[0]
        # find next pow of 2 which is greater or eq to n
        pow_of_2 = 2 ** (np.ceil(np.log2(n)))

        align_filter_dB = np.array([[0, -200], [50, -40], [100, -20],
                                    [125, -12], [160, -6], [200, 0],
                                    [250, 4], [300, 6], [350, 8], [400, 10],
                                    [500, 11], [600, 12], [700, 12], [800, 12],
                                    [1000, 12], [1300, 12], [1600, 12], [2000, 12],
                                    [2500, 12], [3000, 12], [3250, 12], [3500, 4],
                                    [4000, -200], [5000, -200], [6300, -200],
                                    [8000, -200]])
        print('align filter dB shape: ', align_filter_dB.shape)
        num_of_points, trivial = align_filter_dB.shape
        overallGainFilter = interp1d(align_filter_dB[:, 0], align_filter[:, 1],
                                     1000)

        x = np.zeros((pow_of_2))
        x[:data.shape[0]] = data

        x_fft = fftw.fft(x, pow_of_2)

        freq_resolution = srate / pow_of_2

        factorDb = interp1d(align_filter_dB[:, 0],
                            align_filter_dB[:, 1],
                            list(range(0, (pow_of_2 / 2) + 1) * \
                                 freq_resolution)) - \
                   overallGainFilter
        factor = 10 ** (factorDb / 20)

        factor = [factor, np.fliplr(factor[1:(pow_of_2 / 2 + 1)])]
        x_fft = x_fft * factor

        y = fftw.ifft(x_fft, pow_of_2)

        data_filtered = y[:n]
        return data_filtered

    def asl_P56(self, x, srate, nbits):
        """ ITU P.56 method B. """
        T = 0.03  # time constant of smoothing in seconds
        H = 0.2  # hangover time in seconds
        M = 15.9

        # margin in dB of the diff b/w threshold and active speech level
        thres_no = nbits - 1  # num of thresholds, for 16 bits it's 15

        I = np.ceil(srate * H)  # hangover in samples
        g = np.exp(-1 / (srate * T))  # smoothing factor in envelop detection
        c = 2. ** (np.array(list(range(-15, (thres_no + 1) - 16))))
        # array of thresholds from one quantizing level up to half the max
        # code, at a step of 2. In case of 16bit: from 2^-15 to 0.5
        a = np.zeros(c.shape[0])  # activity counter for each level thres
        hang = np.ones(c.shape[0]) * I  # hangover counter for each level thres

        assert x.ndim == 1, x.shape
        sq = np.dot(x, x)  # long term level square energy of x
        x_len = x.shape[0]

        # use 2nd order IIR filter to detect envelope q
        x_abs = np.abs(x)
        p = lfilter(np.ones(1) - g, np.array([1, -g]), x_abs)
        q = lfilter(np.ones(1) - g, np.array([1, -g]), p)

        for k in range(x_len):
            for j in range(thres_no):
                if q[k] >= c[j]:
                    a[j] = a[j] + 1
                    hang[j] = 0
                elif hang[j] < I:
                    a[j] = a[j] + 1
                    hang[j] = hang[j] + 1
                else:
                    break
        asl = 0
        asl_ms = 0
        c0 = None
        if a[0] == 0:
            return asl_ms, asl, c0
        else:
            den = a[0] + self.eps
            AdB1 = 10 * np.log10(sq / a[0] + self.eps)

        CdB1 = 20 * np.log10(c[0] + self.eps)
        if AdB1 - CdB1 < M:
            return asl_ms, asl, c0
        AdB = np.zeros(c.shape[0])
        CdB = np.zeros(c.shape[0])
        Delta = np.zeros(c.shape[0])
        AdB[0] = AdB1
        CdB[0] = CdB1
        Delta[0] = AdB1 - CdB1

        for j in range(1, AdB.shape[0]):
            AdB[j] = 10 * np.log10(sq / (a[j] + self.eps) + self.eps)
            CdB[j] = 20 * np.log10(c[j] + self.eps)

        for j in range(1, Delta.shape[0]):
            if a[j] != 0:
                Delta[j] = AdB[j] - CdB[j]
                if Delta[j] <= M:
                    # interpolate to find the asl
                    asl_ms_log, cl0 = self.bin_interp(AdB[j],
                                                      AdB[j - 1],
                                                      CdB[j],
                                                      CdB[j - 1],
                                                      M, 0.5)
                    asl_ms = 10 ** (asl_ms_log / 10)
                    asl = (sq / x_len) / asl_ms
                    c0 = 10 ** (cl0 / 20)
                    break
        return asl_ms, asl, c0

    def bin_interp(self, upcount, lwcount, upthr, lwthr, Margin, tol):
        if tol < 0:
            tol = -tol

        # check if extreme counts are not already the true active value
        iterno = 1
        if np.abs(upcount - upthr - Margin) < tol:
            asl_ms_log = lwcount
            cc = lwthr
            return asl_ms_log, cc
        if np.abs(lwcount - lwthr - Margin) < tol:
            asl_ms_log = lwcount
            cc = lwthr
            return asl_ms_log, cc

        midcount = (upcount + lwcount) / 2
        midthr = (upthr + lwthr) / 2
        # repeats loop until diff falls inside tolerance (-tol <= diff <= tol)
        while True:
            diff = midcount - midthr - Margin
            if np.abs(diff) <= tol:
                break
            # if tol is not met up to 20 iters, then relax tol by 10%
            iterno += 1
            if iterno > 20:
                tol *= 1.1

            if diff > tol:
                midcount = (upcount + midcount) / 2
                # upper and mid activities
                midthr = (upthr + midthr) / 2
                # ... and thresholds
            elif diff < -tol:
                # then new bounds are...
                midcount = (midcount - lwcount) / 2
                # middle and lower activities
                midthr = (midthr + lwthr) / 2
                # ... and thresholds
        # since tolerance has been satisfied, midcount is selected as
        # interpolated value with tol [dB] tolerance
        asl_ms_log = midcount
        cc = midthr
        return asl_ms_log, cc

    def __repr__(self):
        attrs = '(noises_dir={}\n, snr_levels={}\n, do_IRS={})'.format(
            self.noises_dir,
            self.snr_levels,
            self.do_IRS
        )
        return self.__class__.__name__ + attrs

class Whisperize(object):

    def __init__(self, sr=16000, cache_dir=None, report=False):
        self.report = report
        self.sr = 16000
        self.AHOCODE = 'ahocoder16_64 $infile $f0file $ccfile $fvfile'
        self.AHODECODE = 'ahodecoder16_64 $f0file $ccfile $fvfile $outfile'
        self.cache_dir = cache_dir

    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        if 'uttname' in pkg:
            # look for the uttname in whisper format first
            wuttname = os.path.basename(pkg['uttname'])
        if self.cache_dir is not None and \
                os.path.exists(self.cache_dir) and 'uttname' in pkg:
            wfpath = os.path.join(self.cache_dir, wuttname)
            if not os.path.exists(wfpath):
                raise ValueError('Path {} does not exist'.format(wfpath))
            # The cached whisper file exists, load it and chunk it
            # to match pkg boundaries
            wav, rate = sf.read(wfpath)
            beg_i = pkg['chunk_beg_i']
            end_i = pkg['chunk_end_i']
            L_ = end_i - beg_i
            if len(wav) < L_:
                P = L_ - len(wav)
                wav = np.concatenate((wav, np.zeros((P,))), axis=0)
            assert end_i - beg_i <= len(wav), len(wav)
            wav = wav[beg_i:end_i]
        else:
            wav = wav.data.numpy().reshape(-1).astype(np.float32)
            tf = tempfile.NamedTemporaryFile()
            tfname = tf.name
            # save wav to file
            infile = tfname + '.wav'
            ccfile = tfname + '.cc'
            f0file = tfname + '.lf0'
            fvfile = tfname + '.fv'
            # overwrite infile
            outfile = infile
            inwav = np.array(wav).astype(np.float32)
            # save wav
            sf.write(infile, wav, self.sr)
            # encode with vocoder
            ahocode = self.AHOCODE.replace('$infile', infile)
            ahocode = ahocode.replace('$f0file', f0file)
            ahocode = ahocode.replace('$fvfile', fvfile)
            ahocode = ahocode.replace('$ccfile', ccfile)
            p = subprocess.Popen(shlex.split(ahocode))
            p.wait()
            # read vocoder to know the length
            lf0 = read_aco_file(f0file, (-1,))
            nsamples = lf0.shape[0]
            # Unvoice everything generating -1e10 for logF0 and 
            # 1e3 for FV params
            lf0 = -1e10 * np.ones(nsamples)
            fv = 1e3 * np.ones(nsamples)
            # Write the unvoiced frames overwriting voiced ones
            write_aco_file(fvfile, fv)
            write_aco_file(f0file, lf0)
            # decode with vododer
            ahodecode = self.AHODECODE.replace('$f0file', f0file)
            ahodecode = ahodecode.replace('$ccfile', ccfile)
            ahodecode = ahodecode.replace('$fvfile', fvfile)
            ahodecode = ahodecode.replace('$outfile', outfile)
            p = subprocess.Popen(shlex.split(ahodecode))
            p.wait()
            wav, _ = sf.read(outfile)
            wav = norm_energy(wav.astype(np.float32), inwav)
            if len(wav) > len(inwav):
                wav = wav[:len(inwav)]
            tf.close()
            os.unlink(infile)
            os.unlink(ccfile)
            os.unlink(f0file)
            os.unlink(fvfile)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['whisper'] = True
        pkg['chunk'] = torch.FloatTensor(wav)
        return pkg


    def __repr__(self):
        attrs = '(cache_dir={})'.format(self.cache_dir)
        return self.__class__.__name__ + attrs



class Codec2Buffer(object):

    def __init__(self, kbps=1600, sr=16000, report=False):
        import pycodec2
        self.report = report
        self.kbps = kbps
        self.sr = sr
        self.c2 = pycodec2.Codec2(kbps)
        self.INT16_BYTE_SIZE = 2
        self.FRAME_SIZE = self.c2.samples_per_frame()
        #self.PACKET_SIZE = self.c2.samples_per_frame() * self.INT16_BYTE_SIZE
        #self.STRUCT_FORMAT = '{}h'.format(self.c2.samples_per_frame())

    #@profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        tensor_wav = wav
        re_factor = self.sr // 8000
        wav = wav.data.numpy().reshape(-1).astype(np.float32)
        inwav = wav
        #wav = resample(wav, len(wav) // re_factor)
        #wav = librosa.core.resample(wav, self.sr, 8000,
        #                            res_type='kaiser_fast')
        wav = decimate(wav, self.sr // 8000)
        wav = np.array(wav * (2 ** 15), dtype=np.int16)
        total_frames = int(np.ceil(len(wav) / self.FRAME_SIZE))
        P_ = total_frames * self.FRAME_SIZE - len(wav)
        orilen = len(wav)
        if P_ > 0:
            wav = np.concatenate((wav, 
                                  np.zeros((P_,), dtype=np.int16)),
                                 axis=0)
        owav = []
        T = len(wav)
        data = [wav[t:t + self.FRAME_SIZE] for t in range(0, T,
                                                          self.FRAME_SIZE)]
        for frame in data:
            enc = self.c2.encode(frame)
            dec = self.c2.decode(enc)
            owav.extend(dec.tolist())
        owav = np.array(owav, dtype=np.int16)
        owav = owav[:orilen]
        #owav = np.array(owav, dtype=np.float32) / (2 ** 15)
        # resample up to original srate
        owav = resample(owav, len(owav) * re_factor)
        #owav = librosa.core.resample(owav, 8000, self.sr, res_type='kaiser_fast')
        owav = owav.astype(np.float32) / (2 ** 15)
        #owav = owav / (2 ** 15)
        owav = norm_energy(owav, inwav)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['kbps'] = self.kbps
        #tensor_wav.data = torch.from_numpy(owav)
        pkg['chunk'] = torch.from_numpy(owav)
        return pkg

    def __repr__(self):
        attrs = '(kbps={})'.format(
            self.kbps
        )
        return self.__class__.__name__ + attrs

class Codec2Cached(object):

    def __init__(self, cache_path, kbps=1600, cache=False):
        self.kbps = kbps
        self.cache_path = cache_path
        self.do_cache = cache
        if cache:
            self.cache = {}
            wavs = glob.glob(os.path.join(self.cache_path, '*.wav'))
            for wav in wavs:
                self.load_file(os.path.basename(wav))
            

    def load_file(self, path):
        if not os.path.exist(uttpath):
            raise FileNotFoundError('Could not find the file {}'
                                    ' in the codec2cache path {}'
                                    ''.format(uttname, self.cache_path))
        if hasattr(self, 'cache') and path in cache:
            return self.cache[path]
        else:
            x, rate = sf.read(path)
            if hasattr(self, 'cache'):
                self.cache[path] = x
            return x

    def __call__(self, pkg):
        pkg = format_package(pkg)
        # take input ref for energy normalization
        inwav = pkg['chunk']
        uttname = os.path.basename(pkg['uttname'])
        uttpath = os.path.join(self.cache_path,
                               uttname)
        owav = self.load_file(uttpath)
        owav = norm_energy(owav, inwav)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['kbps'] = self.kbps
        pkg['chunk'] = torch.from_numpy(owav)
        self.do_cache = cache
        return pkg

    def __repr__(self):
        attrs = '(kbps={}, cache_dir={}, cache={})'.format(
            self.cache_dir,
            self.do_cache,
            self.kbps
        )
        return self.__class__.__name__ + attrs


class Codec2(object):

    def __init__(self, kbps=1600, sr=16000, report=False):
        self.kbps = kbps
        self.report = report
        self.sr = sr
        self.SOX_ENCODE = 'sox $infile -r 8k -b 16 -e signed-integer -c 1 $raw_efile'
        self.C2_ENCODE = 'c2enc $kbps $raw_efile $c2file'
        self.C2_DECODE = 'c2dec $kbps $c2file $raw_dfile'
        self.SOX_DECODE = 'sox -r 8k -b 16 -e signed-integer -c 1 $raw_dfile -r $sr -b 16 -e signed-integer -c 1 $outfile'

    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy().reshape(-1).astype(np.float32)
        tf = tempfile.NamedTemporaryFile()
        tfname = tf.name
        # save wav to file
        infile = tfname + '.wav'
        raw_efile = tfname + '.raw'
        c2file = tfname + '.c2'
        raw_dfile = tfname + '_out.raw'
        outfile = tfname + '_out.wav'
        inwav = np.array(wav).astype(np.float32)
        # save wav
        sf.write(infile, wav, self.sr)
        # convert to proper codec 2 input format as raw data with SoX
        sox_encode = self.SOX_ENCODE.replace('$infile', infile)
        sox_encode = sox_encode.replace('$raw_efile', raw_efile)
        p = subprocess.Popen(shlex.split(sox_encode),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.wait()
        #print(sox_encode)
        # Encode with Codec 2
        c2_encode = self.C2_ENCODE.replace('$kbps', str(self.kbps))
        c2_encode = c2_encode.replace('$raw_efile', raw_efile)
        c2_encode = c2_encode.replace('$c2file', c2file)
        #print(c2_encode)
        p = subprocess.Popen(shlex.split(c2_encode))
        p.wait()
        # Decode with Codec 2
        c2_decode = self.C2_DECODE.replace('$kbps', str(self.kbps))
        c2_decode = c2_decode.replace('$c2file', c2file)
        c2_decode = c2_decode.replace('$raw_dfile', raw_dfile)
        #print(c2_decode)
        p = subprocess.Popen(shlex.split(c2_decode), 
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.wait()
        # Convert back to <sr> sampling rate wav
        sox_decode = self.SOX_DECODE.replace('$raw_dfile', raw_dfile)
        sox_decode = sox_decode.replace('$sr', str(self.sr))
        sox_decode = sox_decode.replace('$outfile', outfile)
        #print(sox_decode)
        p = subprocess.Popen(shlex.split(sox_decode),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.wait()
        wav, rate = sf.read(outfile)
        wav = norm_energy(wav.astype(np.float32), inwav)
        tf.close()
        os.unlink(infile)
        os.unlink(raw_efile)
        os.unlink(c2file)
        os.unlink(raw_dfile)
        os.unlink(outfile)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['kbps'] = self.kbps
        pkg['chunk'] = torch.FloatTensor(wav)
        return pkg

    def __repr__(self):
        attrs = '(kbps={})'.format(
            self.kbps
        )
        return self.__class__.__name__ + attrs
        
class SpeedChange(object):

    def __init__(self, factor_range=(-0.15, 0.15), report=False):
        self.factor_range = factor_range
        self.report = report

    # @profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy().reshape(-1).astype(np.float32)
        warp_factor = random.random() * (self.factor_range[1] - \
                                         self.factor_range[0]) + \
                      self.factor_range[0]
        samp_warp = wav.shape[0] + int(warp_factor * wav.shape[0])
        rwav = signal.resample(wav, samp_warp)
        if len(rwav) > len(wav):
            mid_i = (len(rwav) // 2) - len(wav) // 2
            rwav = rwav[mid_i:mid_i + len(wav)]
        if len(rwav) < len(wav):
            diff = len(wav) - len(rwav)
            P = (len(wav) - len(rwav)) // 2
            if diff % 2 == 0:
                rwav = np.concatenate((np.zeros(P, ),
                                       wav,
                                       np.zeros(P, )),
                                      axis=0)
            else:
                rwav = np.concatenate((np.zeros(P, ),
                                       wav,
                                       np.zeros(P + 1, )),
                                      axis=0)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['warp_factor'] = warp_factor
        pkg['chunk'] = torch.FloatTensor(rwav)
        return pkg

    def __repr__(self):
        attrs = '(factor_range={})'.format(
            self.factor_range
        )
        return self.__class__.__name__ + attrs

if __name__ == '__main__':
    """
    lpc = Gammatone(n_channels=40, f_min=100)
    wav, size = sf.read('test.wav')
    wav = wav[:32000]
    print(wav.shape)
    gtn = lpc({'chunk':wav})['gtn'].data.numpy()
    lps = LPS()({'chunk':torch.FloatTensor(wav)})['lps'].data.numpy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    plt.imshow(np.log(gtn))
    plt.subplot(2,1,2)
    #plt.imshow(lps)
    #plt.subplot(3,1,3)
    plt.plot(wav)
    plt.tight_layout()
    plt.savefig('gtn.png', dpi=200)
    """
    import json
    dist_path = '/home/santi/DB/GEnhancement/distortions_SEGANnoises.cfg'
    dtr = json.load(open(dist_path, 'r'))
    dist = config_distortions(**dtr)
    codec = Codec2Buffer()
    wav, size = sf.read('test.wav')
    for n in range(100):
        #buffer_c2 = dist({'chunk':torch.tensor(wav)})['chunk']
        buffer_c2 = codec({'chunk':torch.tensor(wav)})['chunk']
    sf.write('/tmp/buffer_test.wav', buffer_c2, 16000)
