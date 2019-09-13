import torch
import torch.nn as nn
from ..frontend import WaveFe
from ..modules import *
import torch.nn.functional as F
import json
import random
from pase.utils import *
import sys

def minion_maker(cfg):
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = json.load(f)
    print("=" * 50)
    print("name", cfg["name"])
    print("=" * 50)
    mtype = cfg.pop('type', 'mlp')
    if mtype == 'mlp':
        minion = MLPMinion(**cfg)
    elif mtype == 'decoder':
        minion = DecoderMinion(**cfg)
    elif mtype == 'wavernn':
        minion = WaveRNNMinion(**cfg)
    elif mtype == 'spc':
        minion = SPCMinion(**cfg)
    elif mtype == 'gap':
        minion = GapMinion(**cfg)
    elif mtype == 'gru':
        minion = GRUMinion(**cfg)
    elif mtype == 'regularizer':
        minion = RegularizerMinion(**cfg)
    else:
        raise TypeError('Unrecognized minion type {}'.format(mtype))
    return minion

class RegularizerMinion(object):


    def __init__(self, num_inputs=None,
                 loss='MSELoss',
                 loss_weight=1.,
                 name=''):
        if isinstance(loss, str):
            self.loss = getattr(nn, loss)()
        else:
            self.loss = loss
        self.loss_weight = loss_weight
        self.name = name

    def __call__(self, x, alpha=1, device=None):
        return self.forward(x, alpha=alpha, device=device)

    def forward(self, x, alpha=1, device=None):
        # identity function
        return x

class WaveRNNMinion(Model):

    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, num_inputs,
                 rnn_dims=512,
                 fc_dims=512,
                 bits=9,
                 sample_rate=16000,
                 hop_length=160,
                 mode='RAW',
                 pad=2,
                 upsample_cfg={},
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 name='WaveRNNMinion'):
        super().__init__(name=name)
        feat_dims = num_inputs
        self.num_inputs = num_inputs
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW':
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError("Unknown model mode value - ", self.mode)

        upsample_cfg['feat_dims'] = num_inputs
        upsample_cfg['pad'] = pad
        self.upsample = UpsampleNetwork(**upsample_cfg)

        self.rnn_dims = rnn_dims
        self.aux_dims = self.upsample.num_outputs // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

        if keys is None:
            keys = [name]
        self.sg = ScaleGrad()

    def forward(self, x, mels, alpha=1, device=None):
        self.sg.apply(x, alpha)
        device = next(self.parameters()).device  # use same device as parameters
        
        self.step += 1
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        h2 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def generate(self, mels, save_path, batched, target, overlap, mu_law):
        device = next(self.parameters()).device  # use same device as parameters

        mu_law = mu_law if self.mode == 'RAW' else False

        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():

            mels = torch.as_tensor(mels, device=device)
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims, device=device)
            h2 = torch.zeros(b_size, self.rnn_dims, device=device)
            x = torch.zeros(b_size, 1, device=device)

            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                a1_t, a2_t, a3_t, a4_t = \
                    (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.mode == 'MOL':
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    # x = torch.FloatTensor([[sample]]).cuda()
                    x = sample.transpose(0, 1)

                elif self.mode == 'RAW':
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.mode)

                #if i % 100 == 0: self.gen_display(i, seq_len, b_size, start)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)

        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out

        save_wav(output, save_path)

        self.train()

        return output


    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side='both'):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c, device=x.device)
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):

        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()
        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)
        Details:
            x = [[h1, h2, ... hn]]
            Where each h is a vector of conditioning features
            Eg: target=2, overlap=1 with x.size(1)=10
            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = torch.zeros(num_folds, target + 2 * overlap, features, device=x.device)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def xfade_and_unfold(self, y, target, overlap):

        ''' Applies a crossfade and unfolds into a 1d array.
        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64
        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]
            Apply a gain envelope at both ends of the sequences
            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]
            Stagger and add up the groups of samples:
            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]
        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded


class DecoderMinion(Model):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, 
                 dropout_time=0.0,
                 shuffle = False,
                 shuffle_depth = 7,
                 hidden_size=256,
                 hidden_layers=2,
                 fmaps=[256, 256, 128, 128, 128, 64, 64],
                 strides=[2, 2, 2, 2, 2, 5],
                 kwidths=[2, 2, 2, 2, 2, 5],
                 norm_type=None,
                 skip=False,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 name='DecoderMinion'):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.dropout_time = dropout_time
        self.shuffle = shuffle
        self.shuffle_depth = shuffle_depth
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.fmaps = fmaps
        self.strides = strides
        self.kwidths = kwidths
        self.norm_type = norm_type
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys

        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        # First go through deconvolving structure
        for (fmap, kw, stride) in zip(fmaps, kwidths, strides):
            block = GDeconv1DBlock(ninp, fmap, kw, stride,
                                   norm_type=norm_type)
            self.blocks.append(block)
            ninp = fmap

        for _ in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size, dropout))
            ninp = hidden_size
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        
        self.sg.apply(x, alpha)
        
        # The following part of the code drops out some time steps, but the worker should reconstruct all of them (i.e, the original signal)
        # This way we encourage learning features with a larger contextual information
        if self.dropout_time > 0:
            mask=(torch.FloatTensor(x.shape[0],x.shape[2]).to('cuda').uniform_() > self.dropout_time).float().unsqueeze(1)
            x=x*mask

        # The following function (when active) shuffles the time order of the input PASE features. Note that the shuffle has a certain depth (shuffle_depth). 
        # This allows shuffling features that are reasonably close, hopefully encouraging PASE to learn a longer context.
        if self.shuffle:
            x = torch.split(x, self.shuffle_depth, dim=2)
            shuffled_x=[]
            for elem in x:
                    r=torch.randperm(elem.shape[2])
                    shuffled_x.append(elem[:,:,r])

            x=torch.cat(shuffled_x,dim=2)

        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h_ = h
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y


class MLPMinion(Model):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, dropout_time=0.0,hidden_size=256,
                 dropin=0.0,
                 hidden_layers=2,
                 context=1,
                 tie_context_weights=False,
                 skip=True,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 augment=False,
                 r=1, 
                 name='MLPMinion',
                 ratio_fixed=None, range_fixed=None, 
                 dropin_mode='std', drop_channels=False, emb_size=100):
        super().__init__(name=name)
        # Implemented with Conv1d layers to not
        # transpose anything in time, such that
        # frontend and minions are attached very simply
        self.num_inputs = num_inputs
        assert context % 2 != 0, context
        self.context = context
        self.tie_context_weights = tie_context_weights
        self.dropout = dropout
        self.dropout_time = dropout_time
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        if keys is None:
            keys = [name]
        # r frames predicted at once in the output
        self.r = r
        # multiplies number of output dims
        self.num_outputs = num_outputs * r
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        for hi in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size,
                                        din=dropin,
                                        dout=dropout,
                                        context=context,
                                        tie_context_weights=tie_context_weights,
                                        emb_size=emb_size, 
                                        dropin_mode=dropin_mode,
                                        range_fixed=range_fixed,
                                        ratio_fixed=ratio_fixed,
                                        drop_channels=drop_channels))
            ninp = hidden_size
            # in case context has been assigned,
            # it is overwritten to 1
            context = 1
        self.W = nn.Conv1d(ninp, self.num_outputs, context,
                           padding=context//2)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        
        if self.dropout_time > 0 and self.context > 1:
            mask=(torch.FloatTensor(x.shape[0],x.shape[2]).to('cuda').uniform_() > self.dropout_time).float().unsqueeze(1)
            x=x*mask

        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y


class GRUMinion(Model):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 skip=True,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 name='GRUMinion'):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        self.rnn = nn.GRU(ninp,
                          hidden_size,
                          num_layers=hidden_layers,
                          batch_first=True,
                          dropout=dropout)
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        h, _ = self.rnn(x.transpose(1, 2))
        h = h.transpose(1, 2)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y


class SPCMinion(MLPMinion):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 ctxt_frames=5,
                 seq_pad=16,
                 skip=True,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 name='SPCMinion'):
        # num_inputs is code dimension in each time-step,
        # so the MLP has [num_inputs x ctxt_frames] inputs
        # as we unroll time dimension to fixed-sized windows
        print('num_inputs: ', num_inputs)
        print('ctxt_frames: ', ctxt_frames)
        num_inputs = (ctxt_frames + 1) * num_inputs
        print('num_inputs: ', num_inputs)
        super().__init__(num_inputs=num_inputs,
                         num_outputs=num_outputs,
                         dropout=dropout,
                         hidden_size=hidden_size,
                         hidden_layers=hidden_layers,
                         skip=skip,
                         loss=loss,
                         loss_weight=loss_weight,
                         keys=keys,
                         name=name)
        self.ctxt_frames = ctxt_frames
        self.seq_pad = seq_pad
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        # x is a batch of sequences
        # of dims [B, channels, time]
        # first select a "central" time-step
        # with enough seq_pad an ctxt_frames
        # margin M = seq_pad + ctxt_frames on both sides
        self.sg.apply(x, alpha)
        seq_pad = self.seq_pad
        N = self.ctxt_frames
        M = seq_pad + N
        idxs_t = list(range(M + 1, x.size(2) - M))
        t = random.choice(idxs_t)

        bsz = x.size(0)

        # now select future_t (to begin future seq)
        idxs_ft = list(range(t + seq_pad, x.size(2) - N))
        future_t = random.choice(idxs_ft)
        idxs_pt = list(range(N, t - seq_pad))
        past_t = random.choice(idxs_pt)

        # chunk input sequences and current frame
        future = x[:, :, future_t:future_t + N].contiguous().view(bsz, -1)
        past = x[:, :, past_t - N:past_t].contiguous().view(bsz, -1)
        current = x[:, :, t].contiguous()

        # positive batch (future data)
        pos = torch.cat((current, future), dim=1)
        # negative batch (past data)
        neg = torch.cat((current, past), dim=1)

        # forward both jointly
        x_full = torch.cat((pos, neg), dim=0).unsqueeze(2)
        h = x_full
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y

class GapMinion(MLPMinion):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, hidden_size=256,
                 hidden_layers=2,
                 skip=True,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 name='GapMinion'):
        super().__init__(num_inputs=num_inputs,
                         num_outputs=num_outputs,
                         dropout=dropout,
                         hidden_size=hidden_size,
                         hidden_layers=hidden_layers,
                         skip=skip,
                         loss=loss,
                         loss_weight=loss_weight,
                         keys=keys,
                         name=name)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        # x is a batch of sequences
        # of dims [B, channels, time]
        # Select randomly two chunks out of T possible
        self.sg.apply(x, alpha)
        T = x.shape[2]
        aidx = torch.LongTensor(np.random.randint(0, T, size=x.shape[0]))
        bidx = torch.LongTensor(np.random.randint(0, T, size=x.shape[0]))
        x_a = []
        x_b = []
        dists = []
        for i_, (aidx_, bidx_) in enumerate(zip(aidx, bidx)):
            x_a.append(x[i_, :, aidx_].unsqueeze(0))
            x_b.append(x[i_, :, bidx_].unsqueeze(0))
            dist = torch.abs(aidx_ - bidx_) / (T - 1)
            dists.append(dist)
        x_a = torch.cat(x_a, dim=0)
        x_b = torch.cat(x_b, dim=0)
        x_full = torch.cat((x_a, x_b), dim=1).unsqueeze(2)
        dists = torch.LongTensor(dists)
        dists = dists.view(-1, 1, 1)
        
        h = x_full
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        # concat groundtruth to preds
        if self.skip:
            return y, h, dists
        else:
            return y, dists

