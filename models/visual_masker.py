import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VisualMasker(nn.Module):
    """
    VRIS-style encoder.

    Inputs:
        cover  - cover image, shape [B, 3, H, W]
        secret - secret image, shape [B, 3, H, W]

    Output:
        encoded image, shape [B, 3, H, W]
    """

    def __init__(self):
        super().__init__()

        # We concatenate the cover and secret.
        # 3 + 3 = 6 input channels.
        self.encoder = nn.Sequential(
            ConvBlock(6, 32),
            nn.MaxPool2d(2),

            ConvBlock(32, 64),
            nn.MaxPool2d(2),

            ConvBlock(64, 128),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            ConvBlock(64, 64),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(inplace=True),

            ConvBlock(32, 32),

            nn.Conv2d(
                32, 3,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)

        x = self.encoder(x)
        x = self.decoder(x)

        return x