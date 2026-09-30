import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HiddenInsight(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2),

            ConvBlock(32, 64),
            nn.MaxPool2d(2),

            ConvBlock(64, 128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            ConvBlock(64, 64),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            ConvBlock(32, 32),

            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, noisy_encoded):
        x = self.encoder(noisy_encoded)
        x = self.decoder(x)

        return x