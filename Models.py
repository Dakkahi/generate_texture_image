import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, features_g=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 1x1 -> 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 4x4 -> 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 8x8 -> 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 16x16 -> 32x32
            nn.ConvTranspose2d(features_g * 2, img_channels, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Tanh()  # 出力を-1～1に制限
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self._block(img_channels, features_d, 4, 2, 1),  # 64x64 -> 32x32
            self._block(features_d, features_d * 2, 4, 2, 1),  # 32x32 -> 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16 -> 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8 -> 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0),  # 4x4 -> 1x1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)
