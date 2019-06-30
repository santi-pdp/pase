import json
import torch
import torch.nn as nn


def waveminionet_parser(cfg_fname):
    with open(cfg_fname, 'r') as cfg_f:
        cfg_all = json.load(cfg_f)
        # change loss section to select those
        # from nn package
        for i, cfg in enumerate(cfg_all):
            cfg_all[i]['loss'] = getattr(nn, 
                                         cfg_all[i]['loss'])()
        return cfg_all
