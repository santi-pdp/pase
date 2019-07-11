import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .modules import *
except ImportError:
    from modules import *



class EmoDRNLSTM(Model):
    """ Based on https://ieeexplore.ieee.org/document/8682154 
        (Li et al. 2019), without MHA
    """
    def __init__(self, num_inputs, num_outputs, max_ckpts=5, 
                 frontend=None, ft_fe=False, dropout=0,
                 rnn_dropout=0, att=False, att_heads=4,
                 att_dropout=0,
                 name='EmoDRNMHA'):
        super().__init__(max_ckpts=max_ckpts, name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.frontend = frontend
        self.ft_fe = ft_fe
        self.drn = nn.Sequential(
            # first conv block (10, 32), 
            nn.Conv1d(num_inputs, 32, 10),
            # decimating x2
            nn.Conv1d(32, 64, 2, stride=2),
            # first residual blocks (2 resblocks)
            ResBasicBlock1D(64, 64, kwidth=5, att=att,
                            att_heads=att_heads,
                            att_dropout=att_dropout), 
            ResBasicBlock1D(64, 64, kwidth=5, att=att,
                            att_heads=att_heads,
                            att_dropout=att_dropout), 
            # dropout feature maps
            nn.Dropout2d(dropout),
            # decimating x2
            nn.Conv1d(64, 128, 2, stride=2),
            # second residual blocks (2 resblocks)
            ResBasicBlock1D(128, 128, kwidth=5, att=att,
                            att_heads=att_heads,
                            att_dropout=att_dropout), 
            ResBasicBlock1D(128, 128, kwidth=5, att=att,
                            att_heads=att_heads,
                            att_dropout=att_dropout), 
            # dropout feature maps
            nn.Dropout2d(dropout),
            nn.Conv1d(128, 256, 1, stride=1),
            # third residual blocks (2 dilated resblocks)
            ResBasicBlock1D(256, 256, kwidth=3, dilation=2,
                            att=att,
                            att_heads=att_heads,
                            att_dropout=att_dropout), 
            ResBasicBlock1D(256, 256, kwidth=3, dilation=2,
                            att=att,
                            att_heads=att_heads,
                            att_dropout=att_dropout), 
            # dropout feature maps
            nn.Dropout2d(dropout),
            nn.Conv1d(256, 512, 1, stride=1),
            # fourth residual blocks (2 dilated resblocks)
            ResBasicBlock1D(512, 512, kwidth=3, dilation=4,
                            att=att,
                            att_heads=att_heads,
                            att_dropout=att_dropout), 
            ResBasicBlock1D(512, 512, kwidth=3, dilation=4,
                            att=att,
                            att_heads=att_heads,
                            att_dropout=att_dropout), 
            # dropout feature maps
            nn.Dropout2d(dropout)
        )
        # recurrent pooling with 2 LSTM layers
        self.rnn = nn.LSTM(512, 512, num_layers=2, batch_first=True,
                           dropout=rnn_dropout)
        # mlp on top (https://ieeexplore.ieee.org/abstract/document/7366551)
        self.mlp = nn.Sequential(
            nn.Conv1d(512, 200, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(200, 200, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(200, num_outputs, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # input x with shape [B, F, T]
        # FORWARD THROUGH DRN
        # ----------------------------
        if self.frontend is not None:
            x = self.frontend(x)
            if not self.ft_fe:
                x = x.detach()
        x = F.pad(x, (4, 5))
        x = self.drn(x)
        # FORWARD THROUGH RNN
        # ----------------------------
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        xt = torch.chunk(x, x.shape[1], dim=1)
        x = xt[-1].transpose(1, 2)
        # FORWARD THROUGH DNn
        # ----------------------------
        x = self.mlp(x)
        return x

class MLPClassifier(Model):

    def __init__(self, num_inputs,
                 frontend=None,
                 num_spks=None,
                 ft_fe=False,
                 hidden_size=2048,
                 hidden_layers=1,
                 z_bnorm=False,
                 max_ckpts=5,
                 time_pool=False,
                 name='MLP'):
        # 2048 default size raises 5.6M params
        super().__init__(name=name, max_ckpts=max_ckpts)
        self.num_inputs = num_inputs
        self.frontend = frontend
        self.ft_fe = ft_fe
        if ft_fe:
            print('Training the front-end')
        if z_bnorm:
            # apply z-norm to the input
            self.z_bnorm = nn.BatchNorm1d(num_inputs, affine=False)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        layers = [nn.Conv1d(num_inputs, hidden_size, 1),
                  nn.LeakyReLU(),
                  nn.BatchNorm1d(hidden_size)]
        for n in range(1, hidden_layers):
            layers += [nn.Conv1d(hidden_size, hidden_size, 1),
                       nn.LeakyReLU(),
                       nn.BatchNorm1d(hidden_size)]
        layers += [nn.Conv1d(hidden_size, num_spks, 1),
                   nn.LogSoftmax(dim=1)]
        self.model = nn.Sequential(*layers)
        self.time_pool = time_pool

    def forward(self, x):
        if self.frontend is not None:
            x = self.frontend(x)
        h = x
        if self.time_pool:
            h = h.mean(dim=2, keepdim=True)
        if not self.ft_fe:
            h = h.detach()
        if hasattr(self, 'z_bnorm'):
            h = self.z_bnorm(h)
        return self.model(h)

class RNNClassifier(Model):

    def __init__(self, num_inputs,
                 frontend=None,
                 num_spks=None,
                 ft_fe=False,
                 hidden_size=1300,
                 hidden_layers=1,
                 z_bnorm=False,
                 uni=False,
                 return_sequence=False,
                 name='RNN'):
        # 1300 default size raises 5.25M params
        super().__init__(name=name, max_ckpts=1000)
        self.num_inputs = num_inputs
        self.frontend = frontend
        self.ft_fe = ft_fe
        if ft_fe:
            print('Training the front-end')
        if z_bnorm:
            # apply z-norm to the input
            self.z_bnorm = nn.BatchNorm1d(num_inputs, affine=False)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        if uni:
            hsize = hidden_size
        else:
            hsize = hidden_size // 2
        self.rnn = nn.GRU(num_inputs, hsize,
                          num_layers=hidden_layers,
                          bidirectional=not uni,
                          batch_first=True)
        self.model = nn.Sequential(
            nn.Conv1d(hidden_size, num_spks, 1),
            nn.LogSoftmax(dim=1)
        )
        self.return_sequence = return_sequence
        self.uni = uni

    def forward(self, x):
        if self.frontend is not None:
            x = self.frontend(x)
        h = x
        if not self.ft_fe:
            h = h.detach()
        if hasattr(self, 'z_bnorm'):
            h = self.z_bnorm(h)
        ht, state = self.rnn(h.transpose(1, 2))
        if self.return_sequence:
            ht = ht.transpose(1, 2)
        else:
            if not self.uni:
                # pick last time-step for each dir
                # first chunk feat dim
                bsz, slen, feats = ht.size()
                ht = torch.chunk(ht.view(bsz, slen, 2, feats // 2), 2, dim=2)
                # now select fwd
                ht_fwd = ht[0][:, -1, 0, :].unsqueeze(2)
                ht_bwd = ht[1][:, 0, 0, :].unsqueeze(2)
                ht = torch.cat((ht_fwd, ht_bwd), dim=1)
            else:
                # just last time-step works
                ht = ht[:, -1, :].unsqueeze(2)
        y = self.model(ht)
        return y

if __name__ == '__main__':
    drn = EmoDRNLSTM(1, 4)
    print(drn)
    x = torch.randn(1, 1, 100)
    y = drn(x)
    print('y size: ', y.size())
