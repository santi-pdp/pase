from pase.dataset import WavDataset, DictCollater, uttwav_collater
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from pase.transforms import *
import argparse
from pase.utils import pase_parser
import tqdm
import os


def make_transforms(opts, minions_cfg):
    trans = [ToTensor()]
    znorm = False
    for minion in minions_cfg:
        name = minion['name']
        if name == 'mi' or name == 'cmi' or name == 'spc':
            continue
        elif name == 'lps':
            znorm = True
            trans.append(LPS(opts.nfft, hop=160, win=400))
        elif name == 'mfcc':
            znorm = True
            trans.append(MFCC(hop=160))
        elif name == 'prosody':
            znorm = True
            trans.append(Prosody(hop=160, win=400))
        elif name == 'chunk':
            znorm = True
        else:
            raise TypeError('Unrecognized module \"{}\"'
                            'whilst building transfromations'.format(name))
    if znorm:
        trans.append(ZNorm(opts.stats))
    trans = Compose(trans)
    return trans

def extract_acos(dloader, transform, save_path, split):
    for bidx, batch in tqdm.tqdm(enumerate(dloader, start=1),
                                 total=len(dloader)):
        # transform the wav batch element
        wav, uttname, _ = batch
        uttname = os.path.splitext(os.path.basename(uttname[0]))[0]
        aco = transform(wav.view(-1))
        for k in aco.keys():
            if 'uttname' in k or 'raw' in k or 'chunk' in k:
                continue
            save_dir = os.path.join(save_path, split, k)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            kname = uttname + '.{}'.format(k)
            torch.save(aco[k], os.path.join(save_dir,
                                            kname))

def main(opts):
    minions_cfg = pase_parser(opts.net_cfg)
    trans = make_transforms(opts, minions_cfg)
    # Build Dataset(s) and DataLoader(s)
    dset = WavDataset(opts.data_root, opts.data_cfg, 'train',
                      preload_wav=False,
                      return_uttname=True)
    dloader = DataLoader(dset, batch_size=1,
                         shuffle=True, collate_fn=uttwav_collater,
                         num_workers=opts.num_workers)
    va_dset = WavDataset(opts.data_root, opts.data_cfg,
                         'valid', 
                          preload_wav=False,
                          return_uttname=True)
    va_dloader = DataLoader(va_dset, batch_size=1,
                            shuffle=False, collate_fn=uttwav_collater,
                            num_workers=opts.num_workers)
    extract_acos(dloader, trans, opts.save_path, 'train')
    extract_acos(va_dloader, trans, opts.save_path, 'valid')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='data/LibriSpeech/Librispeech_spkid_sel')
    parser.add_argument('--data_cfg', type=str, 
                        default='data/librispeech_data.cfg')
    parser.add_argument('--stats', type=str, default='data/librispeech_stats.pkl')
    parser.add_argument('--save_path', type=str, default='data/Librispeech/')
    parser.add_argument('--net_cfg', type=str, default='cfg/all.cfg')
    parser.add_argument('--nfft', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=0)
     
    opts = parser.parse_args()
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    main(opts)
