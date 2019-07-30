# from pase.models.core import Waveminionet
import librosa
from pase.models.modules import VQEMA
from pase.dataset import PairWavDataset, DictCollater, MetaWavConcatDataset
from pase.models.WorkerScheduler.trainer import trainer
#from torchvision.transforms import Compose
from pase.transforms import *
from pase.losses import *
from pase.utils import pase_parser, worker_parser
import pase
from torch.utils.data import DataLoader
import torch
import pickle
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import random
torch.backends.cudnn.benchmark = True


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def str2None(v):
    if v.lower() in ('none'):
        return None
    return v

def make_transforms(opts, workers_cfg):
    trans = [ToTensor()]
    keys = ['totensor']
    # go through all minions first to check whether
    # there is MI or not to make chunker
    mi = False
    for type, minions_cfg in workers_cfg.items():
        for minion in minions_cfg:
            if 'mi' in minion['name']:
                mi = True
        if mi:
            trans.append(MIChunkWav(opts.chunk_size, random_scale=opts.random_scale))
        else:
            trans.append(SingleChunkWav(opts.chunk_size, random_scale=opts.random_scale))

    znorm = False
    for type, minions_cfg in workers_cfg.items():
        for minion in minions_cfg:
            name = minion['name']
            if name == 'mi' or name == 'cmi' or name == 'spc' or \
               name == 'overlap' or name == 'gap':
                continue
            elif name == 'lps':
                znorm = True
                trans.append(LPS(opts.nfft, hop=opts.hop, win=opts.win))
            elif name == 'gtn':
                znorm = True
                trans.append(Gammatone(opts.gtn_fmin, opts.gtn_channels, 
                                       hop=opts.hop, win=opts.win))
            elif name == 'lpc':
                znorm = True
                trans.append(LPC(opts.lpc_order, hop=opts.hop,
                                 win=opts.win))
            elif name == 'fbank':
                znorm = True
                trans.append(FBanks(n_filters=opts.fbank_filters, 
                                    n_fft=opts.nfft,
                                    hop=opts.hop,
                                    win=opts.win))
            elif name == 'mfcc':
                znorm = True
                trans.append(MFCC(hop=opts.hop))
            elif name == 'prosody':
                znorm = True
                trans.append(Prosody(hop=opts.hop, win=opts.win))
            elif name == 'chunk' or name == 'cchunk':
                znorm = False
            else:
                raise TypeError('Unrecognized module \"{}\"'
                                'whilst building transfromations'.format(name))
            keys.append(name)
    if znorm:
        trans.append(ZNorm(opts.stats))
        keys.append('znorm')
    if opts.trans_cache is None:
        trans = Compose(trans)
    else:
        trans = CachedCompose(trans, keys, opts.trans_cache)
    return trans

def config_distortions(reverb_irfiles=[], 
                       reverb_fmt='imp',
                       reverb_data_root='.',
                       reverb_p=0.5,
                       overlap_dir=None,
                       overlap_list=None,
                       overlap_snrs=[0, 5, 10],
                       overlap_reverb=False,
                       overlap_p=0.5,
                       noises_dir=None,
                       noises_snrs=[0, 5, 10],
                       noises_p=0.5,
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
                       chop_p=0.5):
    trans = []
    probs = []
    # this can be shared in two different stages of the pipeline
    reverb = Reverb(reverb_irfiles, ir_fmt=reverb_fmt,
                    data_root=reverb_data_root)
    if reverb_p > 0. and len(reverb_irfiles) > 0:
        trans.append(reverb)
        probs.append(reverb_p)
    if overlap_p > 0. and overlap_dir is not None:
        noise_trans = reverb if overlap_reverb else None
        trans.append(SimpleAdditiveShift(overlap_dir, overlap_snrs,
                                         noises_list=overlap_list,
                                         noise_transform=noise_trans))
        probs.append(overlap_p)
    if noises_p > 0. and noises_dir is not None:
        trans.append(SimpleAdditive(noises_dir, noises_snrs))
        probs.append(noises_p)
    if speed_p > 0. and speed_range is not None:
        # speed changer
        trans.append(SpeedChange(speed_range))
        probs.append(speed_p)
    if resample_p > 0. and len(resample_factors) > 0:
        trans.append(Resample(resample_factors))
        probs.append(resample_p)
    if clip_p > 0. and len(clip_factors) > 0:
        trans.append(Clipping(clip_factors))
        probs.append(clip_p)
    if chop_p > 0. and len(chop_factors) > 0:
        trans.append(Chopper(max_chops=max_chops,
                             chop_factors=chop_factors))
        probs.append(chop_p)
    if bandrop_p > 0. and len(bandrop_irfiles) > 0:
        trans.append(BandDrop(bandrop_irfiles,filt_fmt=bandrop_fmt, data_root=bandrop_data_root))
        probs.append(bandrop_p)

    if downsample_p > 0. and len(downsample_irfiles) > 0:
        trans.append(Downsample(downsample_irfiles,filt_fmt=downsample_fmt, data_root=downsample_data_root))
        probs.append(downsample_p)

    if len(trans) > 0:
        return PCompose(trans, probs=probs)
    else:
        return None

def config_zerospeech(noises_dir=None,
                      noises_snrs=[0, 5, 10]):
    trans = SimpleAdditive(noises_dir, noises_snrs)
    return trans

def build_dataset_providers(opts, minions_cfg):

    dr = len(opts.data_root)
    dc = len(opts.data_cfg)

    if dr > 1 or dc > 1:
        assert dr == dc, (
            "Specced at least one repeated option for data_root or data_cfg."
            "This assumes multiple datasets, and their resp configs should be matched."
            "Currently got {} data_root and {} data_cfg options".format(dr, dc)
        )
        if opts.dtrans_cfg is not None and len(opts.dtrans_cfg) > 0:
            assert dr == len(opts.dtrans_cfg), (
                "Spec one dtrans_cfg per data_root (can be the same) or None"
            )
        #make sure defaults for dataset has been properly set
        if len(opts.dataset) < dr:
            print ('Provided fewer dataset options than data_root. Repeating default.')
            for _ in range(len(opts.datasets), dr):
                opts.dataset.append('LibriSpeechSegTupleWavDataset')
        if len(opts.zero_speech_p) < dr:
            print ('Provided fewer zero_speech_p options than data_roots. Repeating default.')
            for _ in range(len(opts.zero_speech_p), dr):
                opts.zero_speech_p.append(0)

    #this is to set default in proper way, as argparse
    #uses whatever is set as default in append mode as
    #initial values (i.e. do not override them)
    if len(opts.dataset) < 1:
        opts.dataset.append('LibriSpeechSegTupleWavDataset')

    #TODO: allow for different base transforms for different datasets
    trans = make_transforms(opts, minions_cfg)
    print(trans)

    dsets, va_dsets = [], []
    for idx in range(dr):
        print ('Preparing dset for {}'.format(opts.data_root[idx]))
        if opts.dtrans_cfg is not None and \
            len(opts.dtrans_cfg) > 0 and \
            str2None(opts.dtrans_cfg[idx]) is not None :
            with open(opts.dtrans_cfg[idx], 'r') as dtr_cfg:
                dtr = json.load(dtr_cfg)
                #dtr['trans_p'] = opts.distortion_p
                dist_trans = config_distortions(**dtr)
                print(dist_trans)
        else:
            dist_trans = None
        if opts.zerospeech_cfg is not None \
            and len(opts.zero_speech_p) > 0 \
              and opts.zero_speech_p[idx] > 0:
            with open(opts.zerospeech_cfg[idx], 'r') as zsp_cfg:
                ztr = json.load(zsp_cfg)
                zp_trans = config_zerospeech(**ztr)
                print(zp_trans)
        else:
            zp_trans = None
        # Build Dataset(s) and DataLoader(s)
        dataset = getattr(pase.dataset, opts.dataset[idx])
        print ('Dataset name {} and opts {}'.format(dataset, opts.dataset[idx]))
        dset = dataset(opts.data_root[idx], opts.data_cfg[idx], 'train',
                       transform=trans,
                       noise_folder=opts.noise_folder,
                       whisper_folder=opts.whisper_folder,
                       distortion_probability=opts.distortion_p,
                       distortion_transforms=dist_trans,
                       zero_speech_p=opts.zero_speech_p[idx],
                       zero_speech_transform=zp_trans,
                       preload_wav=opts.preload_wav,
                       ihm2sdm=opts.ihm2sdm)

        dsets.append(dset)

        if opts.do_eval:
            va_dset = dataset(opts.data_root[idx], opts.data_cfg[idx],
                              'valid', transform=trans,
                              noise_folder=opts.noise_folder,
                              whisper_folder=opts.whisper_folder,
                              distortion_probability=opts.distortion_p,
                              distortion_transforms=dist_trans,
                              zero_speech_p=opts.zero_speech_p[idx],
                              zero_speech_transform=zp_trans,
                              preload_wav=opts.preload_wav,
                              ihm2sdm=opts.ihm2sdm)
            va_dsets.append(va_dset)

    ret = None
    if len(dsets) > 1:
        ret = (MetaWavConcatDataset(dsets), )
        if opts.do_eval:
            ret = ret + (MetaWavConcatDataset(va_dsets), )
    else:
        ret = (dsets[0], )
        if opts.do_eval:
            ret = ret + (va_dsets[0], )

    if opts.do_eval is False or len(va_dsets) == 0:
        ret = ret + (None, )

    return ret

def train(opts):
    CUDA = True if torch.cuda.is_available() and not opts.no_cuda else False
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
    print('Seeds initialized to {}'.format(opts.seed))

    # ---------------------
    # Build Model

    minions_cfg = worker_parser(opts.net_cfg)
    #make_transforms(opts, minions_cfg)
    opts.random_scale = str2bool(opts.random_scale)

    dset, va_dset = build_dataset_providers(opts, minions_cfg)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, collate_fn=DictCollater(),
                         num_workers=opts.num_workers,drop_last=True,
                         pin_memory=False)
    # Compute estimation of bpe. As we sample chunks randomly, we
    # should say that an epoch happened after seeing at least as many
    # chunks as total_train_wav_dur // chunk_size
    bpe = (dset.total_wav_dur // opts.chunk_size) // opts.batch_size
    opts.bpe = bpe
    if opts.do_eval:
        assert va_dset is not None, (
            "Asked to do validation, but failed to build validation set"
        )
        va_dloader = DataLoader(va_dset, batch_size=opts.batch_size,
                                shuffle=True, collate_fn=DictCollater(),
                                num_workers=opts.num_workers,drop_last=True,
                                pin_memory=False)
        va_bpe = (va_dset.total_wav_dur // opts.chunk_size) // opts.batch_size
        opts.va_bpe = va_bpe
    else:
        va_dloader = None
    # fastet lr to MI
    #opts.min_lrs = {'mi':0.001}

    if opts.fe_cfg is not None:
        with open(opts.fe_cfg, 'r') as fe_cfg_f:
            print(fe_cfg_f)
            fe_cfg = json.load(fe_cfg_f)
            print(fe_cfg)
    else:
        fe_cfg = None

    # load config file for attention blocks
    if opts.att_cfg:
        with open(opts.att_cfg) as f:
            att_cfg = json.load(f)
            print(att_cfg)
    else:
        att_cfg = None

    print(str2bool(opts.tensorboard))
    Trainer = trainer(frontend_cfg=fe_cfg,
                      att_cfg=att_cfg,
                      minions_cfg=minions_cfg,
                      cfg=vars(opts),
                      backprop_mode=opts.backprop_mode,
                      lr_mode=opts.lr_mode,
                      tensorboard=str2bool(opts.tensorboard),
                      device=device)
    # print(Trainer.model)
    print('Frontend params: ', Trainer.model.frontend.describe_params())

    Trainer.model.to(device)

    Trainer.train_(dloader, device=device, valid_dataloader=va_dloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', action='append', 
                        default=[])
    parser.add_argument('--data_cfg', action='append', 
                        default=[])
    parser.add_argument('--dtrans_cfg', action='append', default=[],
                        help='Distortion transform to apply, note in case of'
                              'mutliple datasets, provide config multiple times')
    parser.add_argument('--zerospeech_cfg', action='append', default=None)
    parser.add_argument('--zero_speech_p', action='append', type=float,
                        default=[0.0])
    parser.add_argument('--dataset', action='append',
                        default=[],
                        help='Dataset to be used: '
                             '(1) PairWavDataset, '
                             '(2) LibriSpeechSegTupleWavDataset, '
                             '(Def: LibriSpeechSegTupleWavDataset.)'
                             'When used multiple times, datasets get'
                             'concatenated with ConcatDataset')
    parser.add_argument('--stats', type=str,
                        default='data/librispeech_stats.pkl',
                        help='Stats file')

    parser.add_argument('--noise_folder', type=str, default=None)
    parser.add_argument('--whisper_folder', type=str, default=None)
    parser.add_argument('--distortion_p', type=float, default=0.4)
    parser.add_argument('--net_ckpt', type=str, default=None,
                        help='Ckpt to initialize the full network '
                             '(Def: None).')
    parser.add_argument('--net_cfg', type=str,
                        default=None)
    parser.add_argument('--fe_cfg', type=str, default=None)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='ckpt')
    parser.add_argument('--max_ckpts', type=int, default=5)
    parser.add_argument('--trans_cache', type=str,
                        default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--random_scale', type=str, default='False', help="random scaling of noise")
    parser.add_argument('--chunk_size', type=int, default=16000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--nfft', type=int, default=2048)
    parser.add_argument('--fbank_filters', type=int, default=40)
    parser.add_argument('--lpc_order', type=int, default=25)
    parser.add_argument('--gtn_channels', type=int, default=40)
    parser.add_argument('--gtn_fmin', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--fe_opt', type=str, default='Adam')
    parser.add_argument('--min_opt', type=str, default='Adam')
    parser.add_argument('--lrdec_step', type=int, default=30,
                        help='Number of epochs to scale lr (Def: 30).')
    parser.add_argument('--lrdecay', type=float, default=0,
                        help='Learning rate decay factor with '
                             'cross validation. After patience '
                             'epochs, lr decays this amount in '
                             'all optimizers. ' 
                             'If zero, no decay is applied (Def: 0).')
    parser.add_argument('--dout', type=float, default=0.2)
    parser.add_argument('--fe_lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=0.0004)
    parser.add_argument('--z_lr', type=float, default=0.0004)
    parser.add_argument('--rndmin_train', action='store_true',
                        default=False)
    parser.add_argument('--adv_loss', type=str, default='BCE',
                        help='BCE or L2')
    parser.add_argument('--warmup', type=int, default=1000000000,
                        help='Epoch to begin applying z adv '
                             '(Def: 1000000000 to not apply it).')
    parser.add_argument('--zinit_weight', type=float, default=1)
    parser.add_argument('--zinc', type=float, default=0.0002)
    parser.add_argument('--vq_K', type=int, default=50,
                        help='Number of K embeddings in VQ-enc. '
                             '(Def: 50).')
    parser.add_argument('--log_grad_keys', type=str, nargs='+',
                        default=[])
    parser.add_argument('--vq', action='store_true', default=False,
                        help='Do VQ quantization of enc output (Def: False).')
    parser.add_argument('--cchunk_prior', action='store_true', default=False)
    parser.add_argument('--sup_exec', type=str, default=None)
    parser.add_argument('--hop', type=int, default=160)
    parser.add_argument('--win', type=int, default=400)
    parser.add_argument('--sup_freq', type=int, default=1)
    parser.add_argument('--preload_wav', action='store_true', default=False,
                        help='Preload wav files in Dataset (Def: False).')
    parser.add_argument('--cache_on_load', action='store_true', default=False,
                        help='Argument to activate cache loading on the fly '
                             'for the wav files in datasets (Def: False).')

    parser.add_argument('--no_continue', type=str, default="False",help="whether continue the training")
    parser.add_argument('--lr_mode', type=str, default='step', help='learning rate scheduler mode')
    parser.add_argument('--att_cfg', type=str, help='Path to the config file of attention blocks')
    parser.add_argument('--avg_factor', type=float, default=0, help="running average factor for option running_avg for attention")
    parser.add_argument('--att_mode', type=str, help='options for attention block')
    parser.add_argument('--tensorboard', type=str, default='True', help='use tensorboard for logging')
    parser.add_argument('--backprop_mode', type=str, default='base',help='backprop policy can be choose from: [base, select_one, select_half]')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="drop out rate for workers")
    parser.add_argument('--delta', type=float, help="delta for hyper volume loss scheduling")
    parser.add_argument('--temp', type=float, help="temp for softmax or adaptive losss")
    parser.add_argument('--alpha', type=float, help="alpha for adaptive loss")
    parser.add_argument('--att_K', type=int, help="top K indices to select for attention")

    #this one is for AMI/ICSI parallel like datasets, so one can selectively pick sdm chunks 
    parser.add_argument('--ihm2sdm', type=str, default=None,
                            help="Pick random of one of these channels."
                                 "Can be empty or None in which case only"
                                 "ihm channel gets used for chunk and cchunk")

    opts = parser.parse_args()
    opts.ckpt_continue = not str2bool(opts.no_continue)
    if opts.net_cfg is None:
        raise ValueError('Please specify a net_cfg file')

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))
    train(opts)
