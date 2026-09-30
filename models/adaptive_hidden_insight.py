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
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AdaptiveHiddenInsight(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)

        self.noise_condition = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock(64, 64)

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock(32, 32)

        self.output = nn.Conv2d(
            32,
            3,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, noise_factor):

        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.enc3(x)

        if noise_factor.dim() == 1:
            noise_factor = noise_factor.unsqueeze(1)

        condition = self.noise_condition(
            noise_factor
        )

        condition = condition.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        x = x + condition

        x = self.up1(x)

        x = self.dec1(x)

        x = self.up2(x)

        x = self.dec2(x)

        x = self.output(x)

        x = self.sigmoid(x)

        return x

