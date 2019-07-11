import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import random
import pysptk
import os
import librosa
import struct
import glob
import pickle
import soundfile as sf
from scipy import interpolate
from scipy import signal
from scipy.signal import decimate
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.interpolate import interp1d
from torchvision.transforms import Compose
from ahoproc_tools.interpolate import interpolation


def norm_and_scale(wav):
    assert isinstance(wav, torch.Tensor), type(wav)
    wav = wav / torch.max(torch.abs(wav))
    return wav * torch.rand(1)

def format_package(x):
    if not isinstance(x, dict):
        return {'raw':x}
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

    #@profile
    def __call__(self, pkg, ignore_keys=[]):
        pkg = format_package(pkg)
        for k, st in self.stats.items():
            #assert k in pkg, '{} != {}'.format(list(pkg.keys()),
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

    ##@profile
    def __call__(self, tensor):
        x = tensor
        reports = []
        for ti, transf in enumerate(self.transforms):
            if isinstance(self.probs, list):
                prob = self.probs[ti]
            else:
                prob = self.probs
            if random.random() <= prob:
                x = transf(x)
                if len(x) == 2:
                    # get the report
                    x, report = x
                    reports.append(report)
        if self.report:
            return x, reports
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

    def __init__(self, chunk_size, random_scale=True):
        self.chunk_size = chunk_size
        self.random_scale = random_scale

    def assert_format(self, x):
        # assert it is a waveform and pytorch tensor
        assert isinstance(x, torch.Tensor), type(x)
        #assert x.dim() == 1, x.size()

    ##@profile
    def select_chunk(self, wav, ret_bounds=False):
        # select random index
        chksz = self.chunk_size
        if len(wav) <= chksz:
            # padding time
            P = chksz - len(wav)
            chk = F.pad(wav.view(1, 1, -1), (0, P), mode='reflect').view(-1)
            idx = 0
        else:
            #idxs = list(range(wav.size(0) - chksz))
            #idx = random.choice(idxs)
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
        if self.random_scale:
            pkg['chunk'] = norm_and_scale(pkg['chunk'])
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
        return pkg

class LPS(object):

    def __init__(self, n_fft=2048, hop=80,
                 win=320, 
                 device='cpu'):
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.device = device

    #@profile
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
            wav = wav.to(self.device)
            X = torch.stft(wav, self.n_fft,
                           self.hop, self.win)
            X = torch.norm(X, 2, dim=2).cpu()[:, :max_frames]
            pkg['lps'] = 10 * torch.log10(X ** 2 + 10e-20).cpu()
        return pkg

    def __repr__(self):
        attrs = '(n_fft={}, hop={}, win={}'.format(self.n_fft,
                                                   self.hop,
                                                   self.win)
        attrs += ', device={})'.format(self.device)
        return self.__class__.__name__ + attrs

class MFCC(object):

    def __init__(self, n_fft=2048, hop=80, 
                 order=20, sr=16000):
        self.n_fft = n_fft
        self.hop = hop
        self.order = order
        self.sr = 16000

    #@profile
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
            pkg['mfcc'] = mfcc
        else:
            #print(y.dtype)
            mfcc = librosa.feature.mfcc(y, sr=self.sr,
                                        n_mfcc=self.order,
                                        n_fft=self.n_fft,
                                        hop_length=self.hop
                                       )[:, :max_frames]
            pkg['mfcc'] = torch.tensor(mfcc.astype(np.float32))
        return pkg

    def __repr__(self):
        attrs = '(order={}, sr={})'.format(self.order,
                                           self.sr)
        return self.__class__.__name__ + attrs

class Prosody(object):

    def __init__(self, hop=80, win=320, f0_min=60, f0_max=300,
                 sr=16000):
        self.hop = hop
        self.win = win
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sr = sr

    #@profile
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
            pkg['prosody'] = proso
        else:
            # first compute logF0 and voiced/unvoiced flag
            #f0 = pysptk.rapt(wav.astype(np.float32),
            #                 fs=self.sr, hopsize=self.hop,
            #                 min=self.f0_min, max=self.f0_max,
            #                 otype='f0')
            f0 = pysptk.swipe(wav.astype(np.float64),
                              fs=self.sr, hopsize=self.hop,
                              min=self.f0_min,
                              max=self.f0_max,
                              otype='f0')
            #sound = pm.Sound(wav.astype(np.float32), self.sr)
            #f0 = sound.to_pitch(self.hop / 16000).selected_array['frequency']
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
            #assert lf0.min() > 0, lf0.data.numpy()
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
            pkg['prosody'] = proso
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
                 data_root='.'):
        self.ir_files = ir_files
        assert isinstance(ir_files, list), type(ir_files)
        assert len(ir_files) > 0, len(ir_files)
        self.ir_idxs = list(range(len(ir_files)))
        #self.IR, self.p_max = self.load_IR(ir_file, ir_fmt)
        self.ir_fmt = ir_fmt
        self.report = report
        self.data_root = data_root
        self.max_reverb_len = max_reverb_len

    def load_IR(self, ir_file, ir_fmt):
        ir_file = os.path.join(self.data_root, ir_file)
        #print('loading ir_file: ', ir_file)
        if ir_fmt == 'mat':
            IR = loadmat(ir_file, squeeze_me=True, struct_as_record=False)
            IR = IR['risp_imp']
        elif ir_fmt == 'imp' or ir_fmt == 'txt':
            IR = np.loadtxt(ir_file)
        elif ir_fmt == 'npy':
            IR = np.load(ir_file)
        else:
            raise TypeError('Unrecognized IR format: ', ir_fmt)
        IR = IR[:self.max_reverb_len]
        IR = IR / np.abs(np.max(IR))
        p_max = np.argmax(np.abs(IR))
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
        #wav = wav / np.max(np.abs(wav))
        #rev = signal.fftconvolve(wav, IR, mode='full')
        rev = signal.convolve(wav, IR, mode='full').reshape(-1)
        Er = np.dot(rev, rev)
        #rev = rev / np.max(np.abs(rev))
        # IR delay compensation
        rev = self.shift(rev, -p_max)
        Eratio = np.sqrt(Ex / Er)
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

class Scale(object):
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".
    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth
    """

    def __init__(self, factor=2**31):
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

    #@profile
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
                 max_chops=2, report=False):
        # chop factors in seconds (mean, std) per possible chop
        import webrtcvad
        self.chop_factors = chop_factors
        self.max_chops = max_chops
        # create VAD to get speech chunks
        self.vad = webrtcvad.Vad(2)
        # make scalers to norm/denorm
        self.denormalizer = Scale(1. / ((2 ** 15) - 1))
        self.normalizer = Scale((2 ** 15) - 1)
        self.report = report

    #@profile
    def vad_wav(self, wav, srate):
        """ Detect the voice activity in the 16-bit mono PCM wav and return
            a list of tuples: (speech_region_i_beg_sample, center_sample, 
            region_duration)
        """
        if srate != 16000:
            raise ValueError('Sample rate must be 16kHz')
        window_size = 160 # samples
        regions = []
        curr_region_counter = 0
        init = None
        vad = self.vad
        # first run the vad across the full waveform
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

    #@profile
    def chop_wav(self, wav, srate, speech_regions):
        if len(speech_regions) == 0:
            #print('Skipping no speech regions')
            return wav
        chop_factors = self.chop_factors
        # get num of chops to make
        num_chops = list(range(1, self.max_chops + 1))
        chops = np.asscalar(np.random.choice(num_chops, 1))
        # trim it to available regions
        chops = min(chops, len(speech_regions))
        # build random indexes to randomly pick regions, not ordered
        if chops == 1:
            chop_idxs = [0]
        else:
            chop_idxs = np.random.choice(list(range(chops)), chops, 
                                         replace=False)
        chopped_wav = np.copy(wav)
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
            #print('chop_beg: ', chop_beg)
            #print('chop_end: ', chop_end)
            # chop the selected region with computed dur
            chopped_wav[chop_beg:chop_end] = 0
        return chopped_wav

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
        chopped = self.chop_wav(wav, srate, 
                                speech_regions).astype(np.float32)
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
        return self.__class__.__name__ + attrs

class Clipping(object):

    def __init__(self, clip_factors = [0.3, 0.4, 0.5],
                 report=False):
        self.clip_factors = clip_factors
        self.report = report

    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy().astype(np.float32)
        #cf = np.random.choice(self.clip_factors, 1)
        cf = random.choice(self.clip_factors)
        clip = np.maximum(wav, cf * np.min(wav))
        clip = np.minimum(clip, cf * np.max(wav))
        clipT = torch.FloatTensor(clip)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['clip_factor'] = np.asscalar(cf)
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

    def sample_noise(self):
        if len(self.noises) == 1:
            return self.noises[0]
        else:
            idx = np.random.randint(0, len(self.noises))
            #idx = random.choice(self.nidxs)
            return self.noises[idx]

    def load_noise(self, filename):
        nwav, rate = sf.read(filename)
        return nwav

    def compute_SNR_K(self, signal, noise, snr):
        Ex = np.dot(signal, signal)
        En = np.dot(noise, noise)
        K = np.sqrt(Ex / ((10 ** (snr / 10.)) * En))
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
                              mode='reflect').view(-1).data.numpy()
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
        noise = sel_noise[n_beg_i:n_beg_i + T].astype(np.float32)
        if self.noise_transform is not None:
            noise = self.noise_transform({'chunk':torch.FloatTensor(noise)})['chunk']
            noise = noise.data.numpy()
        # apply padding to equal length now
        noise = F.pad(torch.tensor(noise).view(1, 1, -1),
                      (len(wav) - len(noise), 0),
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
                #nwav = wavfile.read(npath)[1]
                nwav = librosa.load(npath, sr=None)[0]
                self.noises.append({'file':npath, 
                                    'data':nwav.astype(np.float32)})
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
                                           list(range(0, (pow_of_2 / 2) + 1) *\
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
        T = 0.03 # time constant of smoothing in seconds
        H = 0.2 # hangover time in seconds
        M = 15.9

        # margin in dB of the diff b/w threshold and active speech level
        thres_no = nbits - 1 # num of thresholds, for 16 bits it's 15

        I = np.ceil(srate * H) # hangover in samples
        g = np.exp( -1 / (srate * T)) # smoothing factor in envelop detection
        c = 2. ** (np.array(list(range(-15, (thres_no + 1) - 16))))
        # array of thresholds from one quantizing level up to half the max
        # code, at a step of 2. In case of 16bit: from 2^-15 to 0.5
        a = np.zeros(c.shape[0]) # activity counter for each level thres
        hang = np.ones(c.shape[0]) * I # hangover counter for each level thres

        assert x.ndim == 1, x.shape
        sq = np.dot(x, x) # long term level square energy of x
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
                    asl = (sq / x_len ) / asl_ms
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
            cc =lwthr
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

class SpeedChange(object):

    def __init__(self, factor_range=(-0.15, 0.15), report=False):
        self.factor_range = factor_range
        self.report = report

    #@profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy().reshape(-1).astype(np.float32)
        warp_factor = random.random() * (self.factor_range[1] - \
                                         self.factor_range[0]) + \
                      self.factor_range[0]
        samp_warp=wav.shape[0] + int(warp_factor*wav.shape[0])
        rwav = signal.resample(wav, samp_warp)
        if len(rwav) > len(wav):
            mid_i = (len(rwav) // 2) - len(wav) // 2
            rwav = rwav[mid_i:mid_i +  len(wav)]
        if len(rwav) < len(wav):
            diff = len(wav) - len(rwav)
            P = (len(wav) - len(rwav)) // 2
            if diff % 2 == 0:
                rwav = np.concatenate((np.zeros(P,),
                                       wav,
                                       np.zeros(P,)),
                                      axis=0)
            else:
                rwav = np.concatenate((np.zeros(P,),
                                       wav,
                                       np.zeros(P + 1,)),
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
    import librosa
    from torchvision.transforms import Compose
    import soundfile as sf
    wav, rate = librosa.load('test.wav', sr=None)
    trans = Compose([
        ToTensor(),
        MIChunkWav(32000),
        #Clipping()
        #Reverb(['/tmp/IR_223971.imp',
        #        '/tmp/IR_225824.imp',
        #        '/tmp/IR_225825.imp'], report=False, ir_fmt='txt'),
        #SimpleAdditiveShift('../data/LibriSpeech_50h/wav_sel_train',
        #                    noise_transform=Reverb(['/tmp/IR_223971.imp',
        #                                            '/tmp/IR_225824.imp'],
        #                                           ir_fmt='txt'))
        #SimpleAdditive(['../data/noise_non_stationary/wavs/',
        #                '../data/noise_non_stationary/wavs_bg/'], [0])
        Chopper()
        #LPS(),
        #MFCC(),
        #Prosody(hop=160)
    ])
    print(trans)
    x = trans({'raw':wav, 'raw_rand':wav})
    print(list(x.keys()))
    sf.write('/tmp/noisy.wav', x['chunk'], 16000)
    beg_i = x['chunk_beg_i']
    end_i = x['chunk_end_i']
    sf.write('/tmp/clean.wav', wav[beg_i:end_i], 16000)
    """
    beg_i = x['chunk_beg_i']
    end_i = x['chunk_end_i']
    print(x['prosody'].shape)
    lf0 = np.exp(x['prosody'][0, :].data.numpy())
    uv = x['prosody'][1, :].data.numpy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.subplot(3,1,1)
    plt.plot(lf0)
    plt.subplot(3,1,2)
    plt.plot(uv)
    plt.subplot(3,1,3)
    plt.plot(x['chunk'].data.numpy())
    plt.savefig('/tmp/lf0.png', dpi=200)
    """
    #x = trans({'raw':wav, 'raw_rand':wav})
    #print(x['chunk'].shape)
    #sf.write('/tmp/speed.wav', x['chunk'].data.numpy(), 16000)
    #sf.write('/tmp/chunk_IR_223971.wav', x['cchunk'], 16000)

