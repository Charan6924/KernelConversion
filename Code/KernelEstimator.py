import torch.nn as nn

# ---------------------------------------------------------------------------
# S2K-style building blocks (single conv per stage, no dropout in backbone)
# ---------------------------------------------------------------------------

class S2KConvBlock(nn.Module):
    """Single Conv2d → BatchNorm2d → LeakyReLU, matching S2K's design."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class S2KEncoderStage(nn.Module):
    """Conv block → MaxPool2d(2). Returns (pre_pool_features, pooled)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = S2KConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


# ---------------------------------------------------------------------------
# KernelEstimator — S2K encoder + global pooling + FC head
#
# Instead of decoding back to a full 512x512 map, the encoder's bottleneck
# is pooled to a global vector and projected to 256 points representing the
# radial profile of the real-valued Fourier-domain kernel.
#
# Spatial map sizes (512x512 input):
# enc0 : 512 → 256 (  1 →  32)
# enc1 : 256 → 128 ( 32 →  64)
# enc2 : 128 →  64 ( 64 → 128)
# enc3 :  64 →  32 (128 → 256)
# enc4 :  32 →  16 (256 → 256)
# enc5 :  16 →   8 (256 → 512)
# enc6 :   8 →   4 (512 → 512)
# enc7 :   4 →   2 (512 → 512)
# bottleneck       @ 2x2 (512 → 512)
# global pool → FC → (B, 256)
#
# The 256-point profile is converted to a 2D radially symmetric kernel
# downstream via grid_sample (bilinear interpolation).
# ---------------------------------------------------------------------------

class KernelEstimator(nn.Module):
    """
    U-Net encoder that predicts a 1D radial kernel profile from PSD input.

    Input : (B, 1, 512, 512) log-normalised PSD
    Output: (B, 256) radial profile of the real-valued Fourier-domain kernel
            (256 equally-spaced samples from normalised frequency 0 to 1)
    """

    def __init__(self):
        super().__init__()

        # ---- Encoder (8 stages) -----------------------------------------
        self.enc0 = S2KEncoderStage(  1,  32) # 512 → 256
        self.enc1 = S2KEncoderStage( 32,  64) # 256 → 128
        self.enc2 = S2KEncoderStage( 64, 128) # 128 → 64
        self.enc3 = S2KEncoderStage(128, 256) #  64 → 32
        self.enc4 = S2KEncoderStage(256, 256) #  32 → 16
        self.enc5 = S2KEncoderStage(256, 512) #  16 → 8
        self.enc6 = S2KEncoderStage(512, 512) #   8 → 4
        self.enc7 = S2KEncoderStage(512, 512) #   4 → 2

        # ---- Bottleneck (2x2) -------------------------------------------
        self.bottleneck = S2KConvBlock(512, 512)

        # ---- Global pooling + FC head -----------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
        )

    def forward(self, psd):
        # -- Encoder ------------------------------------------------------
        _, x = self.enc0(psd) # (B,  32,256,256)
        _, x = self.enc1(x)   # (B,  64,128,128)
        _, x = self.enc2(x)   # (B, 128, 64, 64)
        _, x = self.enc3(x)   # (B, 256, 32, 32)
        _, x = self.enc4(x)   # (B, 256, 16, 16)
        _, x = self.enc5(x)   # (B, 512,  8,  8)
        _, x = self.enc6(x)   # (B, 512,  4,  4)
        _, x = self.enc7(x)   # (B, 512,  2,  2)

        # -- Bottleneck ---------------------------------------------------
        x = self.bottleneck(x) # (B, 512, 2, 2)

        # -- Global pool -> FC -> radial profile --------------------------
        x = self.global_pool(x)   # (B, 512, 1, 1)
        x = self.flatten(x)       # (B, 512)
        profile = self.fc_head(x) # (B, 256)

        return profile
