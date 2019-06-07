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
                 name='EmoDRNMHA'):
        super().__init__(max_ckpts=max_ckpts, name=name)
        self.num_inputs = num_inputs
        self.drn = nn.Sequential(
            # first conv block (10, 32), 
            nn.Conv1d(num_inputs, 32, 10),
            # decimating x2
            nn.Conv1d(32, 64, 2, stride=2),
            # first residual blocks (2 resblocks)
            ResBasicBlock1D(64, 64, kwidth=5),
            ResBasicBlock1D(64, 64, kwidth=5),
            # decimating x2
            nn.Conv1d(64, 128, 2, stride=2),
            # second residual blocks (2 resblocks)
            ResBasicBlock1D(128, 128, kwidth=5),
            ResBasicBlock1D(128, 128, kwidth=5),
            nn.Conv1d(128, 256, 1, stride=1),
            # third residual blocks (2 dilated resblocks)
            ResBasicBlock1D(256, 256, kwidth=3, dilation=2),
            ResBasicBlock1D(256, 256, kwidth=3, dilation=2),
            nn.Conv1d(256, 512, 1, stride=1),
            # fourth residual blocks (2 dilated resblocks)
            ResBasicBlock1D(512, 512,  kwidth=3, dilation=4),
            ResBasicBlock1D(512, 512, kwidth=3, dilation=4)
        )
        # recurrent pooling with 2 LSTM layers
        self.rnn = nn.LSTM(512, 512, num_layers=2, batch_first=True)
        # mlp on top (https://ieeexplore.ieee.org/abstract/document/7366551)
        self.mlp = nn.Sequential(
            nn.Linear(512, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_outputs),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # input x with shape [B, F, T]
        # FORWARD THROUGH DRN
        # ----------------------------
        x = F.pad(x, (4, 5))
        x = self.drn(x)
        # FORWARD THROUGH RNN
        # ----------------------------
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        xt = torch.chunk(x, x.shape[1], dim=1)
        x = xt[-1].squeeze(1)
        # FORWARD THROUGH DNn
        # ----------------------------
        x = self.mlp(x)
        return x

if __name__ == '__main__':
    drn = EmoDRNLSTM(1, 4)
    print(drn)
    x = torch.randn(1, 1, 100)
    y = drn(x)
    print('y size: ', y.size())
