import json
import argparse
import random
from random import shuffle
import numpy as np
import os
import re

def load_filenames(opts):
    rooms = {'smallroom':[], 'mediumroom':[], 'largeroom':[]}
    for room in rooms.keys():
        rir_list_fn = os.path.join(opts.data_root, room, 'rir_list')
        with open(rir_list_fn, 'r') as fn:
            for line in fn:
                rooms[room].append(line.split(' ')[4].strip())
    return rooms

def main(opts):

    if opts.existing_cfg is not None:
        with open(opts.existing_cfg, 'r') as ex_f:
            out = json.load(ex_f)
            out['reverb_data_root'] = opts.data_root
            out['reverb_fmt'] = 'wav'
            out['reverb_irfiles'] = []
    else:
        out = {"reverb_data_root":opts.data_root, 
               "reverb_fmt":"wav",
               "reverb_irfiles":[]}

    rooms = load_filenames(opts)
    final_rirs = []

    rirs = rooms['smallroom']
    if opts.small_room_ratio < 1.0:
        sel = int(len(rirs)*opts.small_room_ratio)
        print ('Found {} in small room. Selecting random {} out of them.'\
                .format(len(rirs), sel))
        shuffle(rirs)
        final_rirs.extend(rirs[:sel])
    else:
        final_rirs.extend(rirs)

    rirs = rooms['mediumroom']
    if opts.medium_room_ratio < 1.0:
        sel = int(len(rirs)*opts.medium_room_ratio)
        print ('Found {} in medium room. Selecting random {} out of them.'\
                .format(len(rirs), sel))
        shuffle(rirs)
        final_rirs.extend(rirs[:sel])
    else:
        final_rirs.extend(rirs)

    rirs = rooms['largeroom']
    if opts.large_room_ratio < 1.0:
        sel = int(len(rirs)*opts.large_room_ratio)
        print ('Found {} in large room. Selecting random {} out of them.'\
                .format(len(rirs), sel))
        shuffle(rirs)
        final_rirs.extend(rirs[:sel])
    else:
        final_rirs.extend(rirs)

    print ('Found total {} rir paths'.format(len(final_rirs)))
    out["reverb_irfiles"].extend(sorted(final_rirs))

    with open(opts.out_file, 'w') as f:
        f.write(json.dumps(out, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--tot_num_rirs', type=int, default=-1)
    parser.add_argument('--small_room_ratio', type=int, default=1.0)
    parser.add_argument('--medium_room_ratio', type=float, default=1.0)
    parser.add_argument('--large_room_ratio', type=float, default=1.0)
    parser.add_argument('--convert2npy', action='store_true', default=False)
    parser.add_argument('--existing_cfg', type=str, default=None)
    parser.add_argument('--out_file', type=str, required=True)

    
    opts = parser.parse_args()
    main(opts)

