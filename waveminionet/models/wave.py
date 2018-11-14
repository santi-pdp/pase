import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .frontend import *
from .minions import *
import numpy as np
import os


class WavmentedNet(BaseModel):

    def __init__(self, opts, num_classes=109):
        super().__init__()
        # augmented wav processing net
        # it trains simultaneously with many tasks
        # forcing a hierarchy of abstraction to distill
        # the contents within waveforms
        do_cuda = opts.cuda
        if opts.pretrained_fe:
            self.wav_fe = torch.load(opts.pretrained_fe)
        else:
            self.wav_fe = WaveFe(fmaps=opts.fe_fmaps,
                                 strides=opts.fe_strides,
                                 kwidths=opts.fe_kwidths,
                                 bnorm=opts.bnorm, 
                                 activation=opts.fe_activation,
                                 q_levels=opts.q_levels,
                                 emb_dim=opts.emb_dim,
                                 bidirectional=opts.fe_bidirectional)
        print('Wavmented frontend: ', self.wav_fe)
        print('-' * 20)
        # -------- MINION STACK -----
        # twice feature maps cause of 2 conv trunks (Either bidi or uni)
        minion_inputs = self.wav_fe.fmaps[-1] * 2
        self.minion_rnn_size = opts.minion_rnn_size
        self.minion_rnn_layers = opts.minion_rnn_layers
        self.minion_dropout = opts.minion_dropout
        self.minion_inputs = minion_inputs
        self.do_minions = opts.minions
        # insert all minions within a sequential stack
        self.minions = nn.ModuleList([])
        if self.do_minions:
            # 1) Spectro minion
            self.n_mels = opts.n_mels
            self.spectro_activation = opts.spectro_activation
            spectro = build_minion(opts.mlp_hidden_minions, minion_inputs,
                                   self.n_mels, self.minion_dropout,
                                   self.minion_rnn_size,
                                   self.minion_rnn_layers,
                                   self.spectro_activation,
                                   bidirectional=opts.minion_bidirectional)
            self.minions.append(spectro)
            # 2) Cepstro minion
            #minion_inputs += self.minion_rnn_size
            self.mfcc_order = opts.mfcc_order
            self.mfcc_activation = opts.mfcc_activation
            cepstro = build_minion(opts.mlp_hidden_minions, minion_inputs,
                                   self.mfcc_order, self.minion_dropout,
                                   self.minion_rnn_size,
                                   self.minion_rnn_layers,
                                   self.mfcc_activation,
                                   bidirectional=opts.minion_bidirectional)
            self.minions.append(cepstro)

            # 3) Proso minion
            self.proso_activation = opts.proso_activation
            # E, lf0, u/v, zcr
            proso = build_minion(opts.mlp_hidden_minions, minion_inputs,
                                 4, self.minion_dropout,
                                 self.minion_rnn_size,
                                 self.minion_rnn_layers,
                                 self.proso_activation,
                                 bidirectional=opts.minion_bidirectional)
            self.minions.append(proso)

        # 4) Word minion
        self.num_classes = num_classes
        self.out_minion = opts.out_minion
        if opts.out_minion == 'cnn':
            word = CNNMinion(minion_inputs,
                             num_classes,
                             self.minion_dropout,
                             kwidths=[4, 4, 4],
                             fmaps=[256, 256, 512],
                             strides=[4, 4, 4],
                             out_activation='LogSoftmax',
                             return_sequence=False)
        elif opts.out_minion == 'fixedcnn':
            word = FixedCNNFCMinion(minion_inputs,
                                    num_classes,
                                    self.minion_dropout,
                                    out_activation='LogSoftmax')
        else:
            word = RNNMinion(minion_inputs,
                             num_classes, 
                             self.minion_rnn_size,
                             self.minion_rnn_layers,
                             self.minion_dropout,
                             out_activation='LogSoftmax',
                             return_sequence=False,
                             bidirectional=opts.minion_bidirectional)
        self.minions.append(word)
        if self.do_minions:
            self.minion_names = ['mel_spec', 'mfcc', 'proso', 'lab']
        else:
            self.minion_names = ['lab']

    def forward(self, x):
        wav = x['wav']
        #print('wav size: ', wav.size())
        if len(wav.size()) == 2:
            # channel dim lacks
            wav = wav.unsqueeze(1)
        if len(wav.size()) == 3 and wav.size(1) > 1:
            # transpose L and C dims for conv
            wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        fe_h = self.wav_fe(wav)
        #print('fe_h size: ', fe_h.size())
        # change to RNN format [B, seqlen, feats]
        fe_h = fe_h.transpose(1 ,2)
        min_input = fe_h
        minion_hs = {}
        carry_h = None
        for m_idx, minion in enumerate(self.minions):
            if m_idx > 0:
                m_h, m_states = minion(min_input + carry_h)
            else:
                m_h, m_states = minion(min_input)
            if self.minion_names[m_idx] == 'proso':
                # U/V MUST be binary output
                m_proso_h = m_h[:, :, :-1]
                m_uv_h = m_h[:, :, -1]
                m_h = torch.cat((m_proso_h, F.sigmoid(m_uv_h)), dim=-1)
            if carry_h is None:
                carry_h = m_states
            else:
                if m_idx + 1 < len(self.minions):
                    carry_h = carry_h + m_states
            minion_hs[self.minion_names[m_idx]] = m_h
        return minion_hs

    def save(self, save_path, epoch, best_val=False):
        model_name = 'e{}-asr.ckpt'.format(epoch)
        if best_val:
            model_name = 'best_val-' + model_name
        save_dict = self.state_dict()
        torch.save(save_dict, os.path.join(save_path, model_name))
        # save the frontend alone as well
        torch.save(self.wav_fe, os.path.join(save_path, 'wav_fe.ckpt'))
