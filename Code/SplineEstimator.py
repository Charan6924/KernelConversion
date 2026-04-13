import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedSplineLayer(nn.Module):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def forward(self, knot_params, batch_size, device):
        deltas = F.softplus(knot_params) + 1e-3
        internal_knots = torch.cumsum(deltas, dim=1)
        internal_knots = internal_knots / (internal_knots[:, -1].unsqueeze(1) + 1e-6)
        #internal_knots = internal_knots * 0.90  
        
        padding = self.degree + 1
        zeros = torch.zeros(batch_size, padding, device=device)
        ones = torch.ones(batch_size, padding, device=device)
        full_knots = torch.cat([zeros, internal_knots, ones], dim=1)

        return full_knots

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv_block(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class KernelEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(1, 32)      # 512 -> 256
        self.enc2 = EncoderBlock(32, 64)     # 256 -> 128
        self.enc3 = EncoderBlock(64, 128)    # 128 -> 64
        self.enc4 = EncoderBlock(128, 256)   # 64 -> 32

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)  # 32x32

        # Decoder path
        self.dec4 = DecoderBlock(512, 256)   # 32 -> 64
        self.dec3 = DecoderBlock(256, 128)   # 64 -> 128
        self.dec2 = DecoderBlock(128, 64)    # 128 -> 256
        self.dec1 = DecoderBlock(64, 32)     # 256 -> 512

        # Final output conv
        self.final_conv = nn.Conv2d(32, 32, 1)

        # Global pooling and FC head for kernel parameters
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 16)
        )

        self.knot_layer = FixedSplineLayer(degree=3)
        self.control_scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, psd):
        # Encoder
        skip1, x = self.enc1(psd)   
        skip2, x = self.enc2(x)     
        skip3, x = self.enc3(x)     
        skip4, x = self.enc4(x)     

        x = self.bottleneck(x)       

        x = self.dec4(x, skip4)     
        x = self.dec3(x, skip3)     
        x = self.dec2(x, skip2)     
        x = self.dec1(x, skip1)      

        # Final conv
        x = self.final_conv(x)       # [B, 32, 512, 512]
        x = self.global_pool(x)      # [B, 32, 1, 1]
        x = self.flatten(x)          # [B, 32]
        raw_out = self.fc_head(x)    # [B, 16]

        raw_control = raw_out[:, :10]
        raw_knots = raw_out[:, 10:]
        control = F.softplus(raw_control, beta=1.0) * self.control_scale

        full_knots = self.knot_layer(raw_knots, control.shape[0], control.device)
        return full_knots, control
