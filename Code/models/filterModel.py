import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.relu(out)


class FilterEstimator(nn.Module):
    """
    Takes paired smooth and sharp PSDs, outputs filter ratios directly.
    Input: smooth_psd [B, 1, 512, 512], sharp_psd [B, 1, 512, 512]
    Output: filter_s2sh [B, 512, 512], filter_sh2s [B, 512, 512]
    """
    def __init__(self, base_channels=32):
        super().__init__()

        # Encoder for concatenated PSDs (2 channels input)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, base_channels, 3, stride=2, padding=1),  # 256x256
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock2D(base_channels),

            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # 128x128
            nn.InstanceNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock2D(base_channels*2),

            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),  # 64x64
            nn.InstanceNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock2D(base_channels*4),
        )

        # Decoder to output filter (2 channels: s2sh and sh2s)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),  # 128x128
            nn.InstanceNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock2D(base_channels*2),

            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),  # 256x256
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            ResBlock2D(base_channels),

            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),  # 512x512
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, 2, 3, padding=1),  # 2 output channels
        )

        # Learnable filter strength
        self.filter_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, smooth_psd, sharp_psd):
        x = torch.cat([smooth_psd, sharp_psd], dim=1)  # [B, 2, 512, 512]

        # Encode
        features = self.encoder(x)  # [B, 128, 64, 64]

        # Decode to filter
        out = self.decoder(features)  # [B, 2, 512, 512]

        filter_s2sh = 1.0 + self.filter_strength * out[:, 0, :, :]  # [B, 512, 512]
        filter_sh2s = 1.0 + self.filter_strength * out[:, 1, :, :]  # [B, 512, 512]

        filter_s2sh = F.softplus(filter_s2sh - 1.0) + 0.1  # min 0.1
        filter_sh2s = F.softplus(filter_sh2s - 1.0) + 0.1

        return filter_s2sh, filter_sh2s

