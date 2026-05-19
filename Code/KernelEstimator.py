import torch
import torch.nn as nn
import torch.nn.functional as F

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


class S2KDecoderStage(nn.Module):
    """ConvTranspose2d upsample → cat skip → single conv block.

    in_ch : channels coming into this decoder stage
    skip_ch : channels of the matching encoder skip connection
    out_ch : output channels
    The conv after concat sees (out_ch + skip_ch) channels.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
        self.conv = S2KConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# KernelEstimator — S2K-aligned backbone adapted for 512×512 input
#
# Spatial map sizes (512×512 input):
# enc0 : 512 → 256 (  1 →  32)
# enc1 : 256 → 128 ( 32 →  64)
# enc2 : 128 →  64 ( 64 → 128)
# enc3 :  64 →  32 (128 → 256)
# enc4 :  32 →  16 (256 → 256)
# enc5 :  16 →   8 (256 → 512)
# enc6 :   8 →   4 (512 → 512)
# enc7 :   4 →   2 (512 → 512)
# bottleneck (512→ 512) @ 2×2
# dec7 :   2 →   4 (512 → 512)
# dec6 :   4 →   8 (512 → 512)
# dec5 :   8 →  16 (512 → 256)
# dec4 :  16 →  32 (256 → 256)
# dec3 :  32 →  64 (256 → 128)
# dec2 :  64 → 128 (128 →  64)
# dec1 : 128 → 256 ( 64 →  32)
# dec0 : 256 → 512 ( 32 →  16)
# final_conv :  16 →   2 (1×1 conv, no activation)
#
# Output: (B, 2, 512, 512)
# channel 0 = real part of the complex Fourier-domain kernel
# channel 1 = imaginary part of the complex Fourier-domain kernel
#
# To recover a torch.complex tensor downstream:
# kernel_complex = torch.complex(out[:, 0], out[:, 1]) # (B, 512, 512)
# ---------------------------------------------------------------------------

class KernelEstimator(nn.Module):
    """
    U-Net generator whose backbone closely follows the S2K architecture
    (Tao et al., NeurIPS 2021), extended to accept 512×512 PSD input.

    Estimates the complex-valued reconstruction kernel in the Fourier domain.

    Input : (B, 1, 512, 512) log-normalised PSD
    Output: (B, 2, 512, 512)
            [:, 0, :, :] — real part of the frequency-domain kernel
            [:, 1, :, :] — imaginary part of the frequency-domain kernel

    To get a complex kernel:
    kernel_complex = torch.complex(out[:, 0], out[:, 1])
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

        # ---- Bottleneck (2×2) -------------------------------------------
        self.bottleneck = S2KConvBlock(512, 512)

        # ---- Decoder (8 stages, mirrors encoder) ------------------------
        self.dec7 = S2KDecoderStage(512, 512, 512) #   2 → 4
        self.dec6 = S2KDecoderStage(512, 512, 512) #   4 → 8
        self.dec5 = S2KDecoderStage(512, 512, 256) #   8 → 16
        self.dec4 = S2KDecoderStage(256, 256, 256) #  16 → 32
        self.dec3 = S2KDecoderStage(256, 256, 128) #  32 → 64
        self.dec2 = S2KDecoderStage(128, 128,  64) #  64 → 128
        self.dec1 = S2KDecoderStage( 64,  64,  32) # 128 → 256
        self.dec0 = S2KDecoderStage( 32,  32,  16) # 256 → 512

        # ---- Output head ------------------------------------------------
        # 1×1 conv: 16 → 2, no activation (outputs are unbounded real values)
        # ch 0 → real part, ch 1 → imaginary part of the complex kernel H(u,v)
        self.final_conv = nn.Conv2d(16, 2, kernel_size=1, bias=True)

    def forward(self, psd):
        # -- Encoder ------------------------------------------------------
        s0, x = self.enc0(psd) # s0:(B,  32,512,512) x:(B,  32,256,256)
        s1, x = self.enc1(x)   # s1:(B,  64,256,256) x:(B,  64,128,128)
        s2, x = self.enc2(x)   # s2:(B, 128,128,128) x:(B, 128, 64, 64)
        s3, x = self.enc3(x)   # s3:(B, 256, 64, 64) x:(B, 256, 32, 32)
        s4, x = self.enc4(x)   # s4:(B, 256, 32, 32) x:(B, 256, 16, 16)
        s5, x = self.enc5(x)   # s5:(B, 512, 16, 16) x:(B, 512, 8, 8)
        s6, x = self.enc6(x)   # s6:(B, 512, 8, 8)   x:(B, 512, 4, 4)
        s7, x = self.enc7(x)   # s7:(B, 512, 4, 4)   x:(B, 512, 2, 2)

        # -- Bottleneck ---------------------------------------------------
        x = self.bottleneck(x) # (B, 512, 2, 2)

        # -- Decoder ------------------------------------------------------
        x = self.dec7(x, s7)   # (B, 512, 4, 4)
        x = self.dec6(x, s6)   # (B, 512, 8, 8)
        x = self.dec5(x, s5)   # (B, 256, 16, 16)
        x = self.dec4(x, s4)   # (B, 256, 32, 32)
        x = self.dec3(x, s3)   # (B, 128, 64, 64)
        x = self.dec2(x, s2)   # (B,  64,128,128)
        x = self.dec1(x, s1)   # (B,  32,256,256)
        x = self.dec0(x, s0)   # (B,  16,512,512)

        # -- Output -------------------------------------------------------
        # (B, 2, 512, 512): ch0=real, ch1=imag of complex kernel H(u,v)
        x = self.final_conv(x)
        return x
