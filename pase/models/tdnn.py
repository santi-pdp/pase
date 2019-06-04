import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .modules import *
except ImportError:
    from modules import *


class StatisticalPooling(nn.Module):

    def forward(self, x):
        # x is 3-D with axis [B, feats, T]
        mu = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return torch.cat((mu, std), dim=1)

class TDNN(Model):
    # Architecture taken from x-vectors extractor
    # https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
    def __init__(self, num_inputs, num_outputs, name='TDNN'):
        super().__init__(name=name)
        self.model = nn.Sequential(
            nn.Conv1d(num_inputs, 512, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1500, 1),
            nn.ReLU(inplace=True),
            StatisticalPooling(),
            nn.Conv1d(3000, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, num_outputs, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    sp = StatisticalPooling()
    x = torch.randn(1, 100, 1000)
    y = sp(x)
    print('y size: ', y.size())
    tdnn = TDNN(120, 1200)
    x = torch.randn(1, 120, 27000)
    y = tdnn(x)
    print('y size: ', y.size())
