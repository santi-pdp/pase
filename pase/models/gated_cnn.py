import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *

#adapted from https://github.com/jojonki/Gated-Convolutional-Networks/blob/master/gated_cnn.py
class GatedCNN(Model):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 seq_len,
                 n_layers,
                 kernel,
                 in_chs,
                 out_chs,
                 res_block_count,
                 ans_size):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv_0 = nn.Conv1d(in_chs, out_chs, kernel)
        # self.b_0 = nn.Parameter(torch.randn(in_chs, out_chs, 1))
        self.conv_gate_0 = nn.Conv1d(in_chs, out_chs, kernel)
        # self.c_0 = nn.Parameter(torch.randn(in_chs, out_chs, 1))

        self.conv = nn.ModuleList([nn.Conv1d(out_chs, out_chs, 1) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList([nn.Conv1d(out_chs, out_chs, 1) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(out_chs, out_chs, 1)) for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(out_chs, out_chs, 1)) for _ in range(n_layers)])

        self.fc = nn.Conv1d(out_chs, out_chs, 1, bias=False)

    def forward(self, x):
        # x: (N, seq_len)

        # Embedding
        bs = x.size(0) # batch size
        channels = x.size(1)
        seq_len = x.size(2)

        # Conv1d
        #    Input : (bs, Cin,  Hin)
        #    Output: (bs, Cout, Hout)
        A = self.conv_0(x)      # (bs, Cout, seq_len)
        # A += self.b_0.repeat(1, channels, seq_len)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len)
        # B += self.c_0.repeat(1, channels, seq_len)
        h = A * F.sigmoid(B)    # (bs, Cout, seq_len)
        res_input = h # TODO this is h1 not h0

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h) #+ self.b[i].repeat(1, 1, seq_len)
            B = conv_gate(h) #+ self.c[i].repeat(1, 1, seq_len)
            h = A * F.sigmoid(B) # (bs, Cout, seq_len)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h

        out = self.fc(h) # (bs, ans_size)
        out = F.log_softmax(out, dim=-1)

        return out

class ResBasicBlock1D(Model):
    """ Adapted from
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, kwidth=3, stride=1,
                 dilation=1, norm_layer=None, name='ResBasicBlock1D'):
        super().__init__(name=name)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        # compute padding given dilation factor
        P  = (kwidth // 2) * dilation
        self.conv1 = nn.Conv1d(inplanes, planes, kwidth,
                               stride=stride, padding=P,
                               bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kwidth,
                               padding=P, dilation=dilation,
                               bias=False)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        identity = self.conv1(x)
        identity = self.bn1(identity)
        identity = self.relu(identity)

        out = self.conv2(identity)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out