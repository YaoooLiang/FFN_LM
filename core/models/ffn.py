__all__ = ['FFN']

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torch.nn as nn

class FilterResponseNormalization(nn.Module):
    def __init__(self, beta, gamma, tau, eps=1e-6):
        super(FilterResponseNormalization, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = torch.Tensor([eps])

    def forward(self, x):

        n, c, h, w = x.shape
        assert (self.gamma.shape[1], self.beta.shape[1], self.tau.shape[1]) == (c, c, c)
        nu2 = torch.mean(x.pow(2), (2,3), keepdims=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma*x + self.beta, self.tau)








class ResBlock(nn.Module):
    def __init__(self, in_channels=32, mid_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=1):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv3d(in_channels, mid_channels, kernel_size, padding=padding)
        self.gn0 = nn.GroupNorm(int(mid_channels/2),mid_channels)
        self.conv1 = nn.Conv3d(mid_channels, out_channels, kernel_size, padding=padding)
        self.gn1 = nn.GroupNorm(int(out_channels/2),out_channels)

    def forward(self, x):
        conv0_out = self.conv0(F.relu(x, inplace=True))
        gn0_out = self.gn0(conv0_out)
        conv1_out = self.conv1(F.relu(gn0_out, inplace=True))
        gn1_out = self.gn1(conv1_out)

        return gn1_out + x


class FFN(nn.Module):
    def __init__(self, in_channels=2, mid_channels=32, out_channels=1, kernel_size=(3, 3, 3), padding=1, depth=12,
                 input_size=[33, 33, 33], delta=[8, 8, 8]):
        super(FFN, self).__init__()

        self.conv0 = nn.Conv3d(in_channels, mid_channels, kernel_size, padding=padding)
        self.gn0 = nn.GroupNorm(int(mid_channels/2),mid_channels)
        self.conv1 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding)
        self.gn1 = nn.GroupNorm(int(mid_channels/2),mid_channels)
        self.resblocks = nn.Sequential(*[ResBlock(mid_channels, mid_channels, mid_channels, kernel_size, padding) for i in range(1, depth)])
        self.conv3 = nn.Conv3d(mid_channels, out_channels, (1, 1, 1))

        self.input_size = np.array(input_size)
        self.delta = np.array(delta)
        self.radii = (self.input_size + self.delta*2) // 2

        self._init_weights()

    def forward(self, x):
        conv0_out = self.conv0(x)
        gn0_out = self.gn0(conv0_out)
        conv1_out = self.conv1(F.relu(gn0_out, inplace=True))
        gn1_out = self.gn1(conv1_out)
        res_out = self.resblocks(gn1_out)
        logits = self.conv3(F.relu(res_out, inplace=True))

        return logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
