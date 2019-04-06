import torch
import torch.nn as nn
from .core import LayerNorm


class AhoCNNEncoder(nn.Module):

    def __init__(self, input_dim, kwidth=3, dropout=0.5, layer_norm=False):
        super().__init__()
        pad = (kwidth - 1) // 2

        if layer_norm:
            norm_layer = LayerNorm
        else:
            norm_layer = nn.BatchNorm1d

        self.enc = nn.Sequential(
            nn.Conv1d(input_dim, 256, kwidth, stride=1, padding=pad),
            norm_layer(256),
            nn.PReLU(256),
            nn.Conv1d(256, 256, kwidth, stride=1, padding=pad),
            norm_layer(256),
            nn.PReLU(256),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(256, 512, kwidth, stride=1, padding=pad),
            norm_layer(512),
            nn.PReLU(512),
            nn.Conv1d(512, 512, kwidth, stride=1, padding=pad),
            norm_layer(512),
            nn.PReLU(512),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(512, 1024, kwidth, stride=1, padding=pad),
            norm_layer(1024),
            nn.PReLU(1024),
            nn.Conv1d(1024, 1024, kwidth, stride=1, padding=pad),
            norm_layer(1024),
            nn.PReLU(1024),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(1024, 1024, kwidth, stride=1, padding=pad),
        )

    def forward(self, x):
        return self.enc(x)


class AhoCNNHourGlassEncoder(nn.Module):

    def __init__(self, input_dim, kwidth=3, dropout=0.5, layer_norm=False):
        super().__init__()
        pad = (kwidth - 1) // 2

        if layer_norm:
            norm_layer = LayerNorm
        else:
            norm_layer = nn.BatchNorm1d

        self.enc = nn.Sequential(
            nn.Conv1d(input_dim, 64, kwidth, stride=1, padding=pad),
            norm_layer(64),
            nn.PReLU(64),
            nn.Conv1d(64, 128, kwidth, stride=1, padding=pad),
            norm_layer(128),
            nn.PReLU(128),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kwidth, stride=1, padding=pad),
            norm_layer(256),
            nn.PReLU(256),
            nn.Conv1d(256, 512, kwidth, stride=1, padding=pad),
            norm_layer(512),
            nn.PReLU(512),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, kwidth, stride=1, padding=pad),
            norm_layer(256),
            nn.PReLU(256),
            nn.Conv1d(256, 128, kwidth, stride=1, padding=pad),
            norm_layer(128),
            nn.PReLU(128),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, kwidth, stride=1, padding=pad),
            norm_layer(64),
            nn.PReLU(64),
        )

    def forward(self, x):
        return self.enc(x)
