import math
import torch
import torch.nn as nn
from .modules import *
import torch.nn.functional as F


class _ASPPModule(Model):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv1d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(Model):
    def __init__(self, inplanes, emb_dim):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, 24, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 24, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 24, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 24, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d((1)),
                                             nn.Conv1d(inplanes, 24, 1, stride=1, bias=False),
                                             nn.BatchNorm1d(24),
                                             nn.ReLU())
        self.conv1 = nn.Conv1d(120, emb_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='linear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class aspp_resblock(Model):

    def __init__(self, in_channel, out_channel):

        super().__init__(name="aspp_resblock")

        self.block1 = nn.Sequential(ASPP(in_channel, out_channel),
                                    nn.Conv1d(out_channel, out_channel, kernel_size=11, stride=1, padding=5, bias=False),
                                    nn.BatchNorm1d(out_channel),
                                    nn.ReLU(out_channel))

        self.block2 = nn.Sequential(ASPP(out_channel, out_channel),
                                    nn.Conv1d(out_channel, out_channel, kernel_size=11, stride=1, padding=5, bias=False),
                                    nn.BatchNorm1d(out_channel),
                                    nn.ReLU(out_channel))

        self._init_weight()

    def forward(self, x):


        out_1 = self.block1(x)

        out_2 = self.block2(out_1)

        y = out_1 + out_2

        return y


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



