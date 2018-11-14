import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
from .frontend import *
from .minions import *
from .encoders import *
import random


class WavSpectroVae(nn.Module):

    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda

        # -------------- Build Conv FE Encoder
        if opts.pretrained_fe:
            self.wav_fe = torch.load(opts.pretrained_fe)
        else:
            self.wav_fe = WaveFe(bnorm=opts.bnorm, fmaps=opts.fe_fmaps)
        print(self.wav_fe.__dict__)
        # get latent dim projection layer
        self.latent_fc = nn.Linear(self.wav_fe.fmaps[-1] * 2, 
                                   opts.latent_dim * 2)
        self.latent_fc = nn.utils.weight_norm(self.latent_fc, 
                                             name='weight')
        # ------------- Speaker ID Embedding
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        self.emb = nn.Embedding(num_trg_spks, opts.latent_dim)

        # ------------- Build Decoder minion
        spectro = MLPMinion(opts.latent_dim, opts.nfft // 2 + 1,
                            opts.minion_dropout, out_activation=None,
                            hidden_size=opts.minion_size,
                            hidden_layers=opts.minion_layers,
                            return_sequence=True)
        self.spectro = spectro


    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        wav = x['wav']
        spk_id = x['lab']
        if len(wav.size()) == 2:
            # channel dim lacks
            wav = wav.unsqueeze(1)
        if len(wav.size()) == 3 and wav.size(1) > 1:
            # transpose L and C dims for conv
            wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        fe_h = self.wav_fe(wav)
        fe_h = fe_h.transpose(1, 2)
        enc_outs = self.latent_fc(F.leaky_relu(fe_h, 0.2))
        # split encoder out into latent params
        mu, logvar = torch.chunk(enc_outs, 2, dim=2)

        # forward spk id through emb
        #spk_id = trg_id.unsqueeze(2)
        spk_id = spk_id.repeat(1, enc_outs.size(1))
        id_emb = self.emb(spk_id)

        z = self.reparameterize(mu, logvar)
        z = z + id_emb
        # decode from z to spectrum
        mag, _ = self.spectro(z)
        return {'mag': mag, 'mu':mu, 'logvar':logvar, 'z':z}

    def parameters(self):
        params = []
        for k, v in self.named_parameters():
            if 'emb.' in k:
                #print('Setting {} l2 wd in emb layer'.format(k))
                params.append({'params':v, 'weight_decay':0.002})
            else:
                #print('k {} NOT l2'.format(k))
                params.append({'params':v})
        return params


class WavSpectroSeqVae(nn.Module):

    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda
        self.do_cuda = do_cuda

        # -------------- Build Conv FE Encoder
        if opts.pretrained_fe:
            self.wav_fe = torch.load(opts.pretrained_fe)
        else:
            self.wav_fe = WaveFe(bnorm=opts.bnorm, fmaps=opts.fe_fmaps)
        print(self.wav_fe.__dict__)
        # get latent dim projection layer
        self.latent_fc = nn.Linear(self.wav_fe.fmaps[-1] * 2, 
                                   opts.latent_dim * 2)
        self.latent_fc = nn.utils.weight_norm(self.latent_fc, 
                                             name='weight')
        # ------------- Speaker ID Embedding
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        self.emb = nn.Embedding(num_trg_spks, opts.latent_dim)
        # wd factor for embedding weights reg (L2)
        self.weight_decay = 0
        # ------------- Build Decoder minion
        self.out_dim = opts.nfft // 2 + 1
        self.minion_size = opts.minion_size
        self.minion_layers = opts.minion_layers
        self.minion_dropout = opts.minion_dropout
        self.rnn_layers = self.minion_layers
        if opts.latent_dim != self.minion_size:
            # adapt diemnsionality with Linear mapping
            self.adapt_fc = nn.Linear(opts.latent_dim, self.minion_size,
                                      bias=False)
        self.decoder = spectrumLM(self.minion_size, self.minion_layers,
                                  self.out_dim, self.minion_dropout,
                                  self.do_cuda)
        """
        self.lstm = nn.LSTM(self.out_dim, self.minion_size,
                            self.minion_layers,
                            batch_first=True,
                            dropout=self.minion_dropout,
                            bidirectional=False)
        self.out_fc = nn.Linear(opts.minion_size, self.out_dim)
        """

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, dec_steps=None,
               dec_cps={}):
        wav = x['wav']
        spk_id = x['lab']
        if len(wav.size()) == 2:
            # channel dim lacks
            wav = wav.unsqueeze(1)
        if len(wav.size()) == 3 and wav.size(1) > 1:
            # transpose L and C dims for conv
            wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        fe_h = self.wav_fe(wav)
        #fe_h = fe_h.transpose(1, 2)
        if dec_steps is None:
            dec_steps = fe_h.size(2)
        # average pooling
        fe_h = F.adaptive_avg_pool1d(fe_h, 1).squeeze(2)
        enc_outs = self.latent_fc(fe_h)
        
        # split encoder out into latent params
        mu, logvar = torch.chunk(enc_outs, 2, dim=1)
        
        # forward spk id through emb and reparam z
        id_emb = self.emb(spk_id).squeeze(1)
        z = self.reparameterize(mu, logvar)
        z = z + id_emb
        if hasattr(self, 'adapt_fc'):
            z = self.adapt_fc(z)

        state = self.init_states(z)
        prev_h = Variable(torch.zeros(z.size(0),
                                      1, self.out_dim))
        if self.do_cuda:
            prev_h = prev_h.cuda()

        frames = []
        # Decode frames
        for t in range(dec_steps):
            if t in dec_cps:
                prev_h = dec_cps[t]
            h, state = self.lstm(prev_h, state)
            prev_h = self.out_fc(h)
            frames.append(prev_h)
        mag = torch.cat(frames, 1)

        return {'mag': mag, 'mu':mu, 'logvar':logvar, 'z':z}

    def init_states(self, var):
        # var is 2-D: [bsz, hidden_dim]
        # turn it into 3-D [layers * directions, bsz, hidden_dim]
        var = var.unsqueeze(0)
        # if there are many layers, replicate
        # initial state
        if self.rnn_layers > 1:
            var = var.repeat(self.rnn_layers, 1, 1)
        return (var, var)

    def parameters(self):
        params = []
        for k, v in self.named_parameters():
            if 'emb.' in k:
                #print('Setting {} l2 wd in emb layer'.format(k))
                params.append({'params':v, 'weight_decay':self.weight_decay})
            else:
                #print('k {} NOT l2'.format(k))
                params.append({'params':v})
        return params

class WavSpectroCNNVae(nn.Module):

    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda
        self.do_cuda = do_cuda

        # -------------- Build Conv FE Encoder
        if opts.pretrained_fe:
            self.wav_fe = torch.load(opts.pretrained_fe)
        else:
            self.wav_fe = WaveFe(bnorm=opts.bnorm, fmaps=opts.fe_fmaps)
        print(self.wav_fe.__dict__)
        # get latent dim projection layer
        self.latent_fc = nn.Linear(self.wav_fe.fmaps[-1] * 2, 
                                   opts.latent_dim * 2)
        self.latent_fc = nn.utils.weight_norm(self.latent_fc, 
                                             name='weight')
        # ------------- Speaker ID Embedding
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        self.emb = nn.Embedding(num_trg_spks, opts.latent_dim)
        # wd factor for embedding weights reg (L2)
        self.weight_decay = 0
        # ------------- Build Decoder minion
        self.out_dim = opts.nfft // 2 + 1
        self.decoder = nn.Sequential(
            nn.Conv1d(self.wav_fe.fmaps[-1] * 2 + opts.latent_dim,
                      32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32, eps=1e-3),
            nn.PReLU(32),
            nn.Conv1d(32,
                      64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, eps=1e-3),
            nn.PReLU(64),
            nn.Conv1d(64,
                      128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.PReLU(128),
            nn.Conv1d(128,
                      64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, eps=1e-3),
            nn.PReLU(64),
            nn.Conv1d(64,
                      32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32, eps=1e-3),
            nn.PReLU(32),
            nn.Conv1d(32,
                      self.out_dim, 1, padding=0, stride=1))
        #self.out_fc = nn.Linear(self.wav_fe.fmaps[-1] * 2, self.out_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, dec_steps=None,
               dec_cps={}):
        wav = x['wav']
        spk_id = x['lab']
        if len(wav.size()) == 2:
            # channel dim lacks
            wav = wav.unsqueeze(1)
        if len(wav.size()) == 3 and wav.size(1) > 1:
            # transpose L and C dims for conv
            wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        fe_h = self.wav_fe(wav)
        #fe_h = fe_h.transpose(1, 2)
        if dec_steps is None:
            dec_steps = fe_h.size(2)
        # average pooling
        avg_fe_h = F.adaptive_avg_pool1d(fe_h, 1).squeeze(2)
        enc_outs = self.latent_fc(avg_fe_h)
        
        # split encoder out into latent params
        mu, logvar = torch.chunk(enc_outs, 2, dim=1)
        
        # forward spk id through emb and reparam z
        id_emb = self.emb(spk_id).squeeze(1)
        z = self.reparameterize(mu, logvar)
        z = z + id_emb
        z = z.unsqueeze(2).repeat(1, 1, fe_h.size(2))
        #print('z size: ', z.size())
        #print('enc_outs size: ', enc_outs.size())
        #print('fe_h size: ', fe_h.size())
        conv_in = torch.cat((fe_h, z), dim=1)
        #print('conv_in size: ', conv_in.size())
        mag = self.decoder(conv_in).transpose(1, 2)

        return {'mag': mag, 'mu':mu, 'logvar':logvar, 'z':z}

    def parameters(self):
        params = []
        for k, v in self.named_parameters():
            if 'emb.' in k:
                #print('Setting {} l2 wd in emb layer'.format(k))
                params.append({'params':v, 'weight_decay':self.weight_decay})
            else:
                #print('k {} NOT l2'.format(k))
                params.append({'params':v})
        return params


class WavSpectroLightCNNVae(nn.Module):

    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda
        self.do_cuda = do_cuda

        # -------------- Build Conv FE Encoder
        if opts.pretrained_fe:
            self.wav_fe = torch.load(opts.pretrained_fe)
        else:
            self.wav_fe = WaveFe(bnorm=opts.bnorm, fmaps=opts.fe_fmaps)
        print(self.wav_fe.__dict__)
        # get latent dim projection layer
        self.latent_fc = nn.Linear(self.wav_fe.fmaps[-1] * 2, 
                                   opts.latent_dim * 2)
        self.latent_fc = nn.utils.weight_norm(self.latent_fc, 
                                             name='weight')
        # ------------- Speaker ID Embedding
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        self.emb = nn.Embedding(num_trg_spks, opts.latent_dim)
        # wd factor for embedding weights reg (L2)
        self.weight_decay = 0
        # ------------- Build Light Decoder minion
        self.out_dim = opts.nfft // 2 + 1
        self.decoder = nn.Sequential(
            nn.Conv1d(self.wav_fe.fmaps[-1] * 2 + opts.latent_dim,
                      512, 3, stride=1, padding=1),
            nn.BatchNorm1d(512, eps=1e-3),
            nn.PReLU(512),
            nn.Conv1d(512,
                      self.out_dim, 3, stride=1, padding=1))
        #self.out_fc = nn.Linear(self.wav_fe.fmaps[-1] * 2, self.out_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, dec_steps=None,
               dec_cps={}):
        wav = x['wav']
        spk_id = x['lab']
        if len(wav.size()) == 2:
            # channel dim lacks
            wav = wav.unsqueeze(1)
        if len(wav.size()) == 3 and wav.size(1) > 1:
            # transpose L and C dims for conv
            wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        fe_h = self.wav_fe(wav)
        #fe_h = fe_h.transpose(1, 2)
        if dec_steps is None:
            dec_steps = fe_h.size(2)
        # average pooling
        avg_fe_h = F.adaptive_avg_pool1d(fe_h, 1).squeeze(2)
        enc_outs = self.latent_fc(avg_fe_h)
        
        # split encoder out into latent params
        mu, logvar = torch.chunk(enc_outs, 2, dim=1)
        
        # forward spk id through emb and reparam z
        id_emb = self.emb(spk_id).squeeze(1)
        z = self.reparameterize(mu, logvar)
        z = z + id_emb
        z = z.unsqueeze(2).repeat(1, 1, fe_h.size(2))
        #print('z size: ', z.size())
        #print('enc_outs size: ', enc_outs.size())
        #print('fe_h size: ', fe_h.size())
        conv_in = torch.cat((fe_h, z), dim=1)
        #print('conv_in size: ', conv_in.size())
        mag = self.decoder(conv_in).transpose(1, 2)

        return {'mag': mag, 'mu':mu, 'logvar':logvar, 'z':z}

    def parameters(self):
        params = []
        for k, v in self.named_parameters():
            if 'emb.' in k:
                #print('Setting {} l2 wd in emb layer'.format(k))
                params.append({'params':v, 'weight_decay':self.weight_decay})
            else:
                #print('k {} NOT l2'.format(k))
                params.append({'params':v})
        return params


class WavAhoLightCNNVae(nn.Module):

    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda
        self.do_cuda = do_cuda

        # -------------- Build Conv FE Encoder
        if opts.pretrained_fe:
            self.wav_fe = torch.load(opts.pretrained_fe)
        else:
            self.wav_fe = WaveFe(bnorm=opts.bnorm, fmaps=opts.fe_fmaps)
        print(self.wav_fe.__dict__)
        # get latent dim projection layer
        self.latent_fc = nn.Linear(self.wav_fe.fmaps[-1] * 2, 
                                   opts.latent_dim * 2)
        self.latent_fc = nn.utils.weight_norm(self.latent_fc, 
                                             name='weight')
        # ------------- Speaker ID Embedding
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        self.emb = nn.Embedding(num_trg_spks, opts.latent_dim)
        # wd factor for embedding weights reg (L2)
        self.weight_decay = 0
        # ------------- Build Light Decoder minion
        self.out_dim = opts.cc_order + 3 # 3: fv, lf0 and uv
        self.decoder = nn.Sequential(
            nn.Conv1d(self.wav_fe.fmaps[-1] * 2 + opts.latent_dim,
                      256, 3, stride=1, padding=1),
            nn.BatchNorm1d(256, eps=1e-3),
            nn.PReLU(256),
            nn.Conv1d(256,
                      128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.PReLU(128),
            nn.Conv1d(128,
                      64, 3, stride=1, padding=1),
            nn.PReLU(64),
            nn.BatchNorm1d(64, eps=1e-3),
            nn.Conv1d(64,
                      self.out_dim, 3, stride=1, padding=1))
        #self.out_fc = nn.Linear(self.wav_fe.fmaps[-1] * 2, self.out_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, dec_steps=None,
               dec_cps={}):
        wav = x['wav']
        spk_id = x['lab']
        if len(wav.size()) == 2:
            # channel dim lacks
            wav = wav.unsqueeze(1)
        if len(wav.size()) == 3 and wav.size(1) > 1:
            # transpose L and C dims for conv
            wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        fe_h = self.wav_fe(wav)
        #fe_h = fe_h.transpose(1, 2)
        if dec_steps is None:
            dec_steps = fe_h.size(2)
        # average pooling
        avg_fe_h = F.adaptive_avg_pool1d(fe_h, 1).squeeze(2)
        enc_outs = self.latent_fc(avg_fe_h)
        
        # split encoder out into latent params
        mu, logvar = torch.chunk(enc_outs, 2, dim=1)
        
        # forward spk id through emb and reparam z
        id_emb = self.emb(spk_id).squeeze(1)
        z = self.reparameterize(mu, logvar)
        z = z + id_emb
        z = z.unsqueeze(2).repeat(1, 1, fe_h.size(2))
        #print('z size: ', z.size())
        #print('enc_outs size: ', enc_outs.size())
        #print('fe_h size: ', fe_h.size())
        conv_in = torch.cat((fe_h, z), dim=1)
        #print('conv_in size: ', conv_in.size())
        aco = self.decoder(conv_in).transpose(1, 2)
        # separate regression and classification
        reg_aco = aco[:, :, :-1]
        cla_aco = F.sigmoid(aco[:, :, -1:])
        return {'reg_aco': reg_aco, 'cla_aco':cla_aco, 
                'mu':mu, 'logvar':logvar, 'z':z}

    def parameters(self):
        params = []
        for k, v in self.named_parameters():
            if 'emb.' in k:
                #print('Setting {} l2 wd in emb layer'.format(k))
                params.append({'params':v, 'weight_decay':self.weight_decay})
            else:
                #print('k {} NOT l2'.format(k))
                params.append({'params':v})
        return params


class WavAhoLightCNNAE(nn.Module):
    """ AutoEncoder, not VAE """
    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda
        self.do_cuda = do_cuda

        # -------------- Build Conv FE Encoder
        if opts.pretrained_fe:
            self.wav_fe = torch.load(opts.pretrained_fe)
        else:
            self.wav_fe = WaveFe(bnorm=opts.bnorm, fmaps=opts.fe_fmaps)

        # ------------- Speaker ID Embedding
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        # ------------- Build Light Decoder minion
        self.out_dim = opts.cc_order + 3 # 3: fv, lf0 and uv
        self.decoder_conv =nn.ModuleList()
        self.decoder_act =nn.ModuleList()
        self.decoder_bn =nn.ModuleList()
        self.decoder_dout = nn.ModuleList()
        fmaps = [1024, 512, 256, 128]
        for n, fmap in enumerate(fmaps):
            if n == 0:
                ninputs = self.wav_fe.fmaps[-1] * 2 + num_trg_spks
            else:
                ninputs = fmaps[n -1] + num_trg_spks
            self.decoder_conv.append(
                nn.Conv1d(ninputs,
                          fmap, 3, stride=1, padding=1),
            )
            self.decoder_bn.append(
                nn.BatchNorm1d(fmap, eps=1e-3),
            )
            self.decoder_act.append(
                nn.PReLU(fmap),
            )
            self.decoder_dout.append(nn.Dropout(0.5))
        self.out_conv = nn.Conv1d(fmaps[-1],
                                  self.out_dim, 3, stride=1, padding=1)
        self.l1_reg_weights = [self.wav_fe.fwd[-1].weight,
                               self.wav_fe.fwd[-1].bias,
                               self.wav_fe.bwd[-1].weight,
                               self.wav_fe.bwd[-1].bias]

    def forward(self, x, dec_steps=None,
               dec_cps={}):
        wav = x['wav']
        spk_id = x['lab']
        if len(wav.size()) == 2:
            # channel dim lacks
            wav = wav.unsqueeze(1)
        if len(wav.size()) == 3 and wav.size(1) > 1:
            # transpose L and C dims for conv
            wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        fe_h = self.wav_fe(wav)
        #fe_h = fe_h.transpose(1, 2)
        if dec_steps is None:
            dec_steps = fe_h.size(2)
        
        spk_oh = Variable(torch.zeros(spk_id.size(0), self.num_trg_spks))
        for bidx in range(spk_id.size(0)):
            spk_oh[bidx, spk_id[bidx].cpu().data[0]] = 1
        spk_oh = spk_oh.view(spk_oh.size(0), -1, 1)
        spk_oh = spk_oh.repeat(1, 1, fe_h.size(-1))
        if self.do_cuda:
            spk_oh = spk_oh.cuda()
        #print('spk_oh size: ', spk_oh.size())
        conv_in = torch.cat((fe_h, spk_oh), dim=1)
        #print('conv_in size: ', conv_in.size())
        #print('conv_in size: ', conv_in.size())
        h = conv_in
        for i, (conv, bn, act) in enumerate(zip(self.decoder_conv,
                                                self.decoder_bn,
                                                self.decoder_act)):
            if i > 0:
                h = torch.cat((h, spk_oh), dim=1)
            h = conv(h)
            h = bn(h)
            h = act(h)
        aco = self.out_conv(h)
        aco = aco.transpose(1, 2).contiguous()
        # separate regression and classification
        reg_aco = aco[:, :, :-1]
        cla_aco = F.sigmoid(aco[:, :, -1:])
        ret = {'reg_aco':reg_aco, 'cla_aco':cla_aco}
        if hasattr(self, 'l1_reg_weights'):
            ret['l1_reg'] = self.l1_reg_weights
        return ret
                

    def parameters(self):
        params = []
        for k, v in self.named_parameters():
            if 'emb.' in k:
                #print('Setting {} l2 wd in emb layer'.format(k))
                params.append({'params':v, 'weight_decay':self.weight_decay})
            else:
                #print('k {} NOT l2'.format(k))
                params.append({'params':v})
        return params


class AhoCNNAE(nn.Module):
    """ AutoEncoder, aho2aho """
    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda
        self.do_cuda = do_cuda
        self.out_dim = opts.cc_order + 3 # 3: fv, lf0 and uv
        self.norm_minmax = opts.norm_minmax
        pad = (opts.kwidth - 1) // 2
        if opts.layer_norm:
            norm_layer = LayerNorm
        else:
            norm_layer = nn.BatchNorm1d
        self.dropout = 0.2 # 20% factor to be left active during test (noise)
        # -------------- Build Conv FE Encoder
        self.enc = AhoCNNEncoder(self.out_dim, opts.kwidth,
                                 dropout=self.dropout, 
                                 layer_norm=opts.layer_norm)
        # -------------- Word recon branch
        self.word_spot = opts.word_spot
        if opts.word_spot:
            self.conv_word = nn.Sequential(
                nn.Conv1d(1024, 1024, opts.kwidth, stride=1, padding=pad),
                norm_layer(1024),
                nn.PReLU(1024),
                nn.MaxPool1d(2),
                nn.Dropout(self.dropout),
                nn.Conv1d(1024, 1024, opts.kwidth, stride=1, padding=pad),
                norm_layer(1024),
                nn.PReLU(1024),
                nn.MaxPool1d(4),
                nn.Dropout(self.dropout)
            )
            if opts.word_rnn_pool:
                self.rnn_word = nn.LSTM(1024, 1024, batch_first=True)
            self.fc_word = nn.Linear(1024, opts.vocab_size)
        # ------------- Speaker ID Embedding
        #self.img_emb = opts.img_emb
        #self.img_dim = opts.img_dim
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        #if self.img_emb:
        #    num_dec_inputs = self.img_dim
        #else:
        num_dec_inputs = self.num_trg_spks
        # ------------- Build Light Decoder minion
        self.decoder_conv = nn.ModuleList()
        self.decoder_act = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        self.decoder_dout = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        fmaps = [512, 512, 256, 256, 128]
        upscales = [1, 2, 1, 2, 2]
        for n, (fmap, upscale) in enumerate(zip(fmaps, upscales)):
            if n == 0:
                ninputs = 1024 + num_dec_inputs
            else:
                ninputs = fmaps[n -1] + num_dec_inputs
            if upscale > 1:
                self.decoder_ups.append(nn.Upsample(scale_factor=(1, upscale),
                                                    mode='bilinear'))
            else:
                self.decoder_ups.append(None)
            self.decoder_conv.append(
                nn.Conv1d(ninputs,
                          fmap, opts.kwidth, stride=1, padding=pad),
            )
            self.decoder_bn.append(
                norm_layer(fmap),
            )
            self.decoder_act.append(
                nn.PReLU(fmap),
            )
            self.decoder_dout.append(nn.Dropout(self.dropout))
        self.out_conv = nn.Conv1d(fmaps[-1],
                                  self.out_dim, opts.kwidth, stride=1, padding=pad)

    def forward(self, x, dec_steps=None,
               dec_cps={}, max_range=None):
        # max_range: possibility to specify maximum range for
        # predicted values
        wav = torch.cat((x['reg_aco'],
                         x['cla_aco']), dim=2)
        spk_id = x['lab']
        # declare dict of return
        ret = {}
        wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        fe_h = self.enc(wav)
         
        if hasattr(self, 'word_spot') and self.word_spot:
            # predict word
            wconv_h = self.conv_word(fe_h)
            wconv_h = wconv_h.transpose(1,2)
            if hasattr(self, 'rnn_word'):
                wrnn_h, states = self.rnn_word(wconv_h)
                ht = states[0].squeeze(0)
            else:
                ht = wconv_h
            wout = self.fc_word(ht)
            ret['word'] = F.sigmoid(wout)
        #if self.img_emb:
            # concat visual features
        #    img_emb = x['img_emb']
        #    print('concatenating imb_emb size: ', img_emb.size())
        #    conv_in = torch.cat((fe_h, img_emb), dim=1)
        #else:
        spk_oh = Variable(torch.zeros(spk_id.size(0), self.num_trg_spks))
        for bidx in range(spk_id.size(0)):
            if len(spk_id.size()) == 3:
                spkid = spk_id[bidx,0].cpu().data[0]
            else:
                spkid = spk_id[bidx].cpu().data[0]
            spk_oh[bidx, spkid] = 1
        spk_oh = spk_oh.view(spk_oh.size(0), -1, 1)
        spk_oh = spk_oh.repeat(1, 1, fe_h.size(-1))
        if self.do_cuda:
            spk_oh = spk_oh.cuda()
        conv_in = torch.cat((fe_h, spk_oh), dim=1)
        #print('conv_in size: ', conv_in.size())
        #print('conv_in size: ', conv_in.size())
        h = conv_in
        for i, (usc, conv, bn, act) in enumerate(zip(self.decoder_ups,
                                                     self.decoder_conv,
                                                     self.decoder_bn,
                                                     self.decoder_act)):
            if i > 0:
                spk_oh = spk_oh.repeat(1, 1, h.size(-1) // spk_oh.size(-1))
                h = torch.cat((h, spk_oh), dim=1)
            if usc is not None:
                h = usc(h.unsqueeze(2)).squeeze(2)
            h = conv(h)
            h = bn(h)
            h = act(h)
        aco = self.out_conv(h)
        aco = aco.transpose(1, 2).contiguous()
        # separate regression and classification
        reg_aco = aco[:, :, :-1]
        if max_range is not None and not self.norm_minmax:
            assert len(max_range) == 2, len(max_range)
            #assert isinstance(max_range[0], torch.Tensor)
            #assert isinstance(max_range[1], torch.Tensor)
            # concat feat dims to compare min and max
            reg_aco = reg_aco.unsqueeze(-1)
            def batch_range(range_T, x):
                # make a batch repeating range into x dims
                assert len(range_T.size()) == 1, range_T.size()
                range_T = range_T.view(1, 1, -1, 1)
                range_T = range_T.repeat(x.size(0), x.size(1), 1, 1)
                return range_T
            # lower bound
            min_r = batch_range(max_range[0], reg_aco)
            concat = torch.cat((reg_aco, min_r), dim=-1)
            reg_aco = torch.max(concat, dim=-1)[0]
            # upper bound
            reg_aco = reg_aco.unsqueeze(-1)
            max_r = batch_range(max_range[1], reg_aco)
            concat = torch.cat((reg_aco, max_r), dim=-1)
            reg_aco = torch.min(concat, dim=-1)[0]
        cla_aco = F.sigmoid(aco[:, :, -1:])
        if hasattr(self, 'norm_minmax') and self.norm_minmax:
            reg_aco = F.tanh(reg_aco)
        ret['reg_aco']= reg_aco
        ret['cla_aco'] = cla_aco
        return ret
                

class AhoCNNHourGlass(AhoCNNAE):
    """ AutoEncoder in HourGlass shape, aho2aho """
    def __init__(self, opts, spks):
        super().__init__(opts, spks)

        do_cuda = opts.cuda
        self.do_cuda = do_cuda
        self.out_dim = opts.cc_order + 3 # 3: fv, lf0 and uv
        self.norm_minmax = opts.norm_minmax
        pad = (opts.kwidth - 1) // 2
        if opts.layer_norm:
            norm_layer = LayerNorm
        else:
            norm_layer = nn.BatchNorm1d
        self.dropout = 0.2 # 20% factor to be left active during test (noise)
        # -------------- Build Conv FE Encoder
        self.enc = AhoCNNHourGlassEncoder(self.out_dim, opts.kwidth,
                                          dropout=self.dropout, 
                                          layer_norm=opts.layer_norm)
        # -------------- Word recon branch
        self.word_spot = opts.word_spot
        if opts.word_spot:
            self.conv_word = nn.Sequential(
                nn.Conv1d(64, 128, opts.kwidth, stride=1, padding=pad),
                norm_layer(128),
                nn.PReLU(128),
                nn.MaxPool1d(2),
                nn.Dropout(self.dropout),
                nn.Conv1d(128, 256, opts.kwidth, stride=1, padding=pad),
                norm_layer(256),
                nn.PReLU(256),
                nn.MaxPool1d(4),
                nn.Dropout(self.dropout)
            )
            if opts.word_rnn_pool:
                self.rnn_word = nn.LSTM(256, 256, batch_first=True)
            self.fc_word = nn.Linear(256, opts.vocab_size)
        # ------------- Speaker ID Embedding
        #self.img_emb = opts.img_emb
        #self.img_dim = opts.img_dim
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        #if hasattr(self, 'img_emb'):
        #    num_dec_inputs = self.img_dim
        #else:
        num_dec_inputs = self.num_trg_spks
        # ------------- Build Light Decoder minion
        self.decoder_conv = nn.ModuleList()
        self.decoder_act = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        self.decoder_dout = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        fmaps = [128, 256, 256, 128, 64]
        upscales = [2, 1, 2, 1, 2]
        for n, (fmap, upscale) in enumerate(zip(fmaps, upscales)):
            if n == 0:
                ninputs = 64 + num_dec_inputs
            else:
                ninputs = fmaps[n -1] + num_dec_inputs
            if upscale > 1:
                self.decoder_ups.append(nn.Upsample(scale_factor=(1, upscale),
                                                    mode='bilinear'))
            else:
                self.decoder_ups.append(None)
            self.decoder_conv.append(
                nn.Conv1d(ninputs,
                          fmap, opts.kwidth, stride=1, padding=pad),
            )
            self.decoder_bn.append(
                norm_layer(fmap),
            )
            self.decoder_act.append(
                nn.PReLU(fmap),
            )
            self.decoder_dout.append(nn.Dropout(self.dropout))
        self.out_conv = nn.Conv1d(fmaps[-1],
                                  self.out_dim, opts.kwidth, stride=1, padding=pad)

class AcoDenoisingAE(nn.Module):
    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda
        self.do_cuda = do_cuda
        self.out_dim = opts.cc_order + 3 # 3: fv, lf0 and uv
        self.norm_minmax = opts.norm_minmax
        # id all layers of decoder
        self.id_all = opts.id_all
        pad = (opts.kwidth - 1) // 2
        #if opts.layer_norm:
        #    norm_layer = LayerNorm
        #else:
        #    norm_layer = nn.BatchNorm1d
        self.dropout = opts.dropout

        # compression type can be: [1:1%, 2:5%, 3:10%, 4:20%, 5:50%, 6:100%]
        # assuming Aco inputs of dim 43
        comp_types = {1:{'fmaps':[40, 20, 10, 4]},
                      2:{'fmaps':[40, 30, 20, 16]},
                      3:{'fmaps':[40, 38, 35, 32]},
                      4:{'fmaps':[40, 45, 50, 64]},
                      5:{'fmaps':[48, 64, 128, 200]},
                      6:{'fmaps':[64, 128, 256, 344]}}

        comp_type = comp_types[opts.comp_type]
        strides = [2, 2, 2, 1]
        # -------------- Build Conv FE Encoder
        self.enc = nn.ModuleList()
        for fi, fmap in enumerate(comp_type['fmaps']):
            if fi == 0:
                ninputs = 43
            else:
                ninputs = comp_type['fmaps'][fi - 1]
            self.enc.append(nn.Conv1d(ninputs, fmap, 4,
                                      stride=strides[fi], padding=1))
            self.enc.append(nn.PReLU(fmap))
            if opts.do_norm:
                self.enc.append(nn.BatchNorm1d(fmap))
            if opts.dropout > 0:
                self.enc.append(nn.Dropout(opts.dropout))
            #if fi < len(comp_type['fmaps']) - 1:
            #    self.enc.append(nn.MaxPool1d(2))
        enc_outs = fmap
        # -------------- Word recon branch
        self.word_spot = opts.word_spot
        if opts.word_spot:
            self.conv_word = nn.Sequential(
                nn.Conv1d(enc_outs, 128, opts.kwidth, stride=2, padding=pad),
                nn.PReLU(128),
                nn.BatchNorm1d(128),
                nn.Dropout(self.dropout),
                nn.Conv1d(128, 256, opts.kwidth, stride=2, padding=pad),
                nn.PReLU(256),
                nn.BatchNorm1d(256),
                nn.Dropout(self.dropout),
                nn.Conv1d(256, 512, opts.kwidth, stride=2, padding=pad),
                nn.PReLU(512),
                nn.BatchNorm1d(512),
                nn.Dropout(self.dropout)
            )
            if opts.word_rnn_pool:
                self.rnn_word = nn.LSTM(512, 256, batch_first=True)
                self.rnn_bn = nn.BatchNorm1d(256)
            self.fc_word = nn.Linear(256, opts.vocab_size)

        # ------------- Speaker ID Embedding
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        num_dec_inputs = self.num_trg_spks
        # ------------- Build Light Decoder minion
        dec_fmaps = comp_type['fmaps'][::-1][1:]
        self.dec = nn.ModuleList()
        for fi, fmap in enumerate(dec_fmaps):
            if fi == 0:
                ninputs = enc_outs + num_trg_spks
            else:
                ninputs = dec_fmaps[fi - 1]
                if self.id_all:
                    ninputs += num_trg_spks
            self.dec.append(nn.ConvTranspose1d(ninputs, fmap, 4,
                                               stride=2, padding=1))
            #self.dec.append(nn.Upsample(scale_factor=(1, 2),
            #                            mode='bilinear'))
            self.dec.append(nn.PReLU(fmap))
            if opts.do_norm:
                self.dec.append(nn.BatchNorm1d(fmap))
            if opts.dropout > 0:
                self.dec.append(nn.Dropout(opts.dropout))
        self.out = nn.Conv1d(fmap, 43, 1, stride=1, padding=0)

    def forward(self, x, dec_steps=None,
               dec_cps={}, max_range=None):
        # max_range: possibility to specify maximum range for
        # predicted values
        wav = torch.cat((x['reg_aco'],
                         x['cla_aco']), dim=2)
        spk_id = x['lab']
        # declare dict of return
        ret = {}
        wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        h = wav
        for enc_l in self.enc:
            h = enc_l(h)
         
        if hasattr(self, 'conv_word'):
            # predict word
            w_h = self.conv_word(h)
            w_h = w_h.transpose(1,2)
            if hasattr(self, 'rnn_word'):
                w_h, states = self.rnn_word(w_h)
                ht = states[0].squeeze(0)
                ht = self.rnn_bn(ht)
            else:
                ht = w_h
            wout = self.fc_word(ht)
            ret['word'] = F.sigmoid(wout)

        spk_oh = Variable(torch.zeros(spk_id.size(0), self.num_trg_spks))
        for bidx in range(spk_id.size(0)):
            if len(spk_id.size()) == 3:
                spkid = spk_id[bidx,0].cpu().data[0]
            else:
                spkid = spk_id[bidx].cpu().data[0]
            spk_oh[bidx, spkid] = 1
        spk_oh = spk_oh.view(spk_oh.size(0), -1, 1)
        if self.do_cuda:
            spk_oh = spk_oh.cuda()
        spk_re_oh = spk_oh.repeat(1, 1, h.size(-1))
        dec_in = torch.cat((h, spk_re_oh), dim=1)
        h = dec_in
        for d_i, dec_l in enumerate(self.dec):
            if self.id_all and d_i > 0 and isinstance(dec_l,
                                                      nn.ConvTranspose1d):
                spk_re_oh = spk_oh.repeat(1, 1, h.size(-1))
                #print('Concating h: {} and spk_re_oh: {}'.format(h.size(),
                #                                                 spk_re_oh.size()))
                h = torch.cat((h, spk_re_oh), dim=1)
            if isinstance(dec_l, nn.Upsample):
                h = dec_l(h.unsqueeze(2)).squeeze(2)
            else:
                h = dec_l(h)
        aco = self.out(h)
        aco = aco.transpose(1, 2).contiguous()
        # separate regression and classification
        reg_aco = aco[:, :, :-1]
        if max_range is not None and not self.norm_minmax:
            assert len(max_range) == 2, len(max_range)
            #assert isinstance(max_range[0], torch.Tensor)
            #assert isinstance(max_range[1], torch.Tensor)
            # concat feat dims to compare min and max
            reg_aco = reg_aco.unsqueeze(-1)
            def batch_range(range_T, x):
                # make a batch repeating range into x dims
                assert len(range_T.size()) == 1, range_T.size()
                range_T = range_T.view(1, 1, -1, 1)
                range_T = range_T.repeat(x.size(0), x.size(1), 1, 1)
                return range_T
            # lower bound
            min_r = batch_range(max_range[0], reg_aco)
            concat = torch.cat((reg_aco, min_r), dim=-1)
            reg_aco = torch.max(concat, dim=-1)[0]
            # upper bound
            reg_aco = reg_aco.unsqueeze(-1)
            max_r = batch_range(max_range[1], reg_aco)
            concat = torch.cat((reg_aco, max_r), dim=-1)
            reg_aco = torch.min(concat, dim=-1)[0]
        cla_aco = F.sigmoid(aco[:, :, -1:])
        if hasattr(self, 'norm_minmax') and self.norm_minmax:
            reg_aco = F.tanh(reg_aco)
        ret['reg_aco']= reg_aco
        ret['cla_aco'] = cla_aco
        return ret
                

class AcoEmbDenoisingAE(nn.Module):
    """ Aco DAE with embeddings enforcing
        spk recognition and word spot.
    """
    def __init__(self, opts, spks):
        super().__init__()

        do_cuda = opts.cuda
        self.do_cuda = do_cuda
        self.out_dim = opts.cc_order + 3 # 3: fv, lf0 and uv
        self.norm_minmax = opts.norm_minmax
        # id all layers of decoder
        self.id_all = opts.id_all
        pad = (opts.kwidth - 1) // 2
        #if opts.layer_norm:
        #    norm_layer = LayerNorm
        #else:
        #    norm_layer = nn.BatchNorm1d
        self.dropout = opts.dropout

        # compression type can be: [1:1%, 2:5%, 3:10%, 4:20%, 5:50%, 6:100%]
        # assuming Aco inputs of dim 43
        comp_types = {1:{'fmaps':[40, 20, 10, 4]},
                      2:{'fmaps':[40, 30, 20, 16]},
                      3:{'fmaps':[40, 38, 35, 32]},
                      4:{'fmaps':[40, 45, 50, 64]},
                      5:{'fmaps':[48, 64, 128, 200]},
                      6:{'fmaps':[64, 128, 256, 344]}}

        comp_type = comp_types[opts.comp_type]
        strides = [2, 2, 2, 1]
        # -------------- Build Conv FE Encoder
        self.enc = nn.ModuleList()
        for fi, fmap in enumerate(comp_type['fmaps']):
            if fi == 0:
                ninputs = 43
            else:
                ninputs = comp_type['fmaps'][fi - 1]
            self.enc.append(nn.Conv1d(ninputs, fmap, opts.kwidth,
                                      stride=strides[fi], padding=1))
            self.enc.append(nn.PReLU(fmap))
            if opts.do_norm and fi < len(comp_type['fmaps']) - 1:
                self.enc.append(nn.BatchNorm1d(fmap))
            if opts.dropout > 0:
                self.enc.append(nn.Dropout(opts.dropout))
        enc_outs = fmap
        self.enc_outs = enc_outs

        # ------------- Speaker ID Embedding + output
        num_trg_spks = len(list(spks.keys()))
        self.num_trg_spks = num_trg_spks
        num_dec_inputs = self.num_trg_spks
        self.emb = nn.Embedding(num_trg_spks, enc_outs)
        self.spk_fc = nn.Linear(enc_outs, num_trg_spks)
        # -------------- Word ID output
        self.word_fc = nn.Linear(enc_outs, opts.vocab_size)

        # ------------- Build Light Decoder minion
        dec_fmaps = comp_type['fmaps'][::-1][1:]
        self.dec = nn.ModuleList()
        for fi, fmap in enumerate(dec_fmaps):
            if fi == 0:
                ninputs = enc_outs
            else:
                ninputs = dec_fmaps[fi - 1]
            self.dec.append(nn.ConvTranspose1d(ninputs, fmap, opts.kwidth,
                                               stride=2, padding=1))
            #self.dec.append(nn.Upsample(scale_factor=(1, 2),
            #                            mode='bilinear'))
            self.dec.append(nn.PReLU(fmap))
            if opts.do_norm:
                self.dec.append(nn.BatchNorm1d(fmap))
            if opts.dropout > 0:
                self.dec.append(nn.Dropout(opts.dropout))
        self.out = nn.Conv1d(fmap, 43, 1, stride=1, padding=0)

    def forward(self, x, dec_steps=None,
                dec_cps={}, max_range=None, phase=0):
        # phase: {0: AE w/ embedding recon.
        #         1: spkID prediction
        #         2: subtract embedding and word prediction

        # max_range: possibility to specify maximum range for
        # predicted values
        wav = torch.cat((x['reg_aco'],
                         x['cla_aco']), dim=2)
        spk_id = x['lab']
        # declare dict of return
        ret = {}
        wav = wav.transpose(1, 2)
        # CNN format [B, feats, seqlen]
        h = wav
        for enc_l in self.enc:
            h = enc_l(h)

        # project ID through embedding
        spk_id = spk_id.view(spk_id.size(0), 1).repeat(1, h.size(2))
        spk_emb = self.emb(spk_id)
        e_h = h.transpose(1,2)
        if phase == 1:
            # predict spkid with shallow output
            spk_out = self.spk_fc(e_h)
            ret['lab'] = F.log_softmax(spk_out)
            return ret
        
        # subtract spk id
        w_h = e_h - spk_emb

        if phase == 2:
            # predict word with shallow output
            ht = self.word_fc(w_h)
            ret['word'] = F.sigmoid(ht)
            return ret

        h = w_h + spk_emb
        h = h.transpose(1, 2)
        # return embedding in case it is analysed
        ret['z'] = h

        for d_i, dec_l in enumerate(self.dec):
            h = dec_l(h)

        aco = self.out(h)
        aco = aco.transpose(1, 2).contiguous()
        # separate regression and classification
        reg_aco = aco[:, :, :-1]
        if max_range is not None and not self.norm_minmax:
            assert len(max_range) == 2, len(max_range)
            #assert isinstance(max_range[0], torch.Tensor)
            #assert isinstance(max_range[1], torch.Tensor)
            # concat feat dims to compare min and max
            reg_aco = reg_aco.unsqueeze(-1)
            def batch_range(range_T, x):
                # make a batch repeating range into x dims
                assert len(range_T.size()) == 1, range_T.size()
                range_T = range_T.view(1, 1, -1, 1)
                range_T = range_T.repeat(x.size(0), x.size(1), 1, 1)
                return range_T
            # lower bound
            min_r = batch_range(max_range[0], reg_aco)
            concat = torch.cat((reg_aco, min_r), dim=-1)
            reg_aco = torch.max(concat, dim=-1)[0]
            # upper bound
            reg_aco = reg_aco.unsqueeze(-1)
            max_r = batch_range(max_range[1], reg_aco)
            concat = torch.cat((reg_aco, max_r), dim=-1)
            reg_aco = torch.min(concat, dim=-1)[0]
        cla_aco = F.sigmoid(aco[:, :, -1:])
        if hasattr(self, 'norm_minmax') and self.norm_minmax:
            reg_aco = F.tanh(reg_aco)
        ret['reg_aco']= reg_aco
        ret['cla_aco'] = cla_aco
        return ret
