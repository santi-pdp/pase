from pase.dataset import LibriSpeechSegTupleWavDataset
from pase.transforms import *

from argparse import ArgumentParser

from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--data_cfg_file", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", help="train or test?", required=False)
    args = parser.parse_args()

    transforms = [ToTensor(), SingleChunkWav(chunk_size=16000), LPS()]
    trans = Compose(transforms)

    dataset = LibriSpeechSegTupleWavDataset(args.data_root, args.data_cfg_file, args.split, trans)

    dl = DataLoader(dataset)
    it = iter(dl)

    try:
        a = next(it)
        while(a != None):
            a = next(it)
            print(a['raw'].shape)
    except StopIteration:
        print("Done")
        exit(0)