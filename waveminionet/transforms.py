import torch
import numpy as np
import random
import pysptk
import librosa
import pickle
from ahoproc_tools.interpolate import interpolation


def norm_and_scale(wav):
    assert isinstance(wav, torch.Tensor), type(wav)
    wav = wav / torch.max(torch.abs(wav))
    return wav * torch.rand(1)

def format_package(x):
    if not isinstance(x, dict):
        return {'raw':x}
    return x

class ToTensor(object):

    def __call__(self, pkg):
        pkg = format_package(pkg)
        for k, v in pkg.items():
            # convert everything in the package
            # into tensors
            if not isinstance(v, torch.Tensor):
                pkg[k] = torch.tensor(v)
        return pkg

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ZNorm(object):

    def __init__(self, stats):
        self.stats_name = stats
        with open(stats, 'rb') as stats_f:
            self.stats = pickle.load(stats_f)

    def __call__(self, pkg):
        pkg = format_package(pkg)
        for k, st in self.stats.items():
            #assert k in pkg, '{} != {}'.format(list(pkg.keys()),
            #                                   list(self.stats.keys()))
            if k in pkg:
                mean = st['mean'].unsqueeze(1)
                std = st['std'].unsqueeze(1)
                pkg[k] = (pkg[k] - mean) / std
        return pkg

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.stats_name)

class SingleChunkWav(object):

    def __init__(self, chunk_size, random_scale=True):
        self.chunk_size = chunk_size
        self.random_scale = random_scale

    def assert_format(self, x):
        # assert it is a waveform and pytorch tensor
        assert isinstance(x, torch.Tensor), type(x)
        assert x.dim() == 1, x.size()

    def select_chunk(self, wav):
        # select random index
        chksz = self.chunk_size
        idxs = list(range(wav.size(0) - chksz))
        idx = random.choice(idxs)
        chk = wav[idx:idx + chksz]
        return chk

    def __call__(self, pkg):
        pkg = format_package(pkg)
        raw = pkg['raw']
        self.assert_format(raw)
        pkg['chunk'] = self.select_chunk(raw)
        if self.random_scale:
            pkg['chunk'] = norm_and_scale(pkg['chunk'])
        return pkg

    def __repr__(self):
        return self.__class__.__name__ + \
                '({})'.format(self.chunk_size)

class MIChunkWav(SingleChunkWav):

    """ Max-Information chunker expects 2 input wavs,
        and extract 3 chunks: (chunk, chunk_ctxt,
        and chunk_rand). The first two correspond to same
        waveform, the third one is sampled from the second wav
    """
    def __call__(self, pkg):
        pkg = format_package(pkg)
        if 'raw_rand' not in pkg:
            raise ValueError('Need an input pair of wavs to do '
                             'MI chunking! Just got single raw wav?')
        raw = pkg['raw']
        raw_rand = pkg['raw_rand']
        self.assert_format(raw)
        self.assert_format(raw_rand)
        pkg['chunk'] = self.select_chunk(raw)
        pkg['chunk_ctxt'] = self.select_chunk(raw)
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

    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        max_frames = wav.size(0) // self.hop
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

    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
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

    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy()
        max_frames = wav.shape[0] // self.hop
        # first compute logF0 and voiced/unvoiced flag
        f0 = pysptk.swipe(wav.astype(np.float64),
                          fs=self.sr, hopsize=self.hop,
                          min=self.f0_min,
                          max=self.f0_max,
                          otype='f0')
        lf0 = np.log(f0 + 1e-10)
        lf0, uv = interpolation(lf0, -1)
        lf0 = torch.tensor(lf0.astype(np.float32)).unsqueeze(0)[:, :max_frames]
        uv = torch.tensor(uv.astype(np.float32)).unsqueeze(0)[:, :max_frames]
        if torch.sum(uv) == 0:
            # if frame is completely unvoiced, make lf0 min val
            lf0 = torch.ones(uv.size()) * np.log(self.f0_min)
        assert lf0.min() > 0, lf0.data.numpy()
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
                

if __name__ == '__main__':
    import librosa
    from torchvision.transforms import Compose
    wav, rate = librosa.load('test.wav')
    trans = Compose([
        ToTensor(),
        MIChunkWav(16000),
        LPS(),
        MFCC(),
        Prosody()
    ])
    print(trans)
    x = trans({'raw':wav, 'raw_rand':wav})
    print(list(x.keys()))



