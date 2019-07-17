from tensorboardX import SummaryWriter
import numpy as np
import torch
import pickle
import os


class PklWriter(object):

    def __init__(self, save_path):
        from datetime import datetime
        curr_time = datetime.now().strftime('%b%d_%H-%M-%S')
        fname = 'losses_{}.pkl'.format(curr_time)
        self.save_path = os.path.join(save_path, fname)
        self.losses = {}

    def add_scalar(self, tag, scalar_value, global_step=None):
        if tag not in self.losses:
            self.losses[tag] = {'global_step':[],
                                'scalar_value':[]}
        if torch.is_tensor(scalar_value):
            scalar_value = scalar_value.item()
        self.losses[tag]['scalar_value'].append(scalar_value)
        self.losses[tag]['global_step'].append(global_step)
        with open(self.save_path, 'wb') as out_f:
            pickle.dump(self.losses, out_f)

    def add_histogram(self, tag, values, global_step=None, bins='sturges'):
        # not implemented for the json logger
        pass

class LogWriter(object):

    def __init__(self, save_path, log_types=['tensorboard', 'pkl']):
        self.save_path = save_path
        if len(log_types) == 0:
            raise ValueError('Please specify at least one log_type file to '
                             'write to in the LogWriter!')
        self.writers = []
        for log_type in log_types:
            if 'tensorboard' == log_type:
                self.writers.append(SummaryWriter(save_path))
            elif 'pkl' == log_type:
                self.writers.append(PklWriter(save_path))
            else:
                raise TypeError('Unrecognized log_writer type: ', log_writer)

    def add_scalar(self, tag, scalar_value, global_step=None):
        for writer in self.writers:
            writer.add_scalar(tag, scalar_value=scalar_value, 
                              global_step=global_step)

    def add_histogram(self, tag, values, global_step=None, bins='sturges'):
        for writer in self.writers:
            writer.add_histogram(tag, values=values, global_step=global_step, 
                                 bins=bins)

