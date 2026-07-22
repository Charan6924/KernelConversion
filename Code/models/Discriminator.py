import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        super().__init__()

        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        #upsampling
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        #downsampling
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(
                ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=4, stride=1, padding=1, bias=False
            ),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, input_nc=1, ndf=64, n_layers=3, num_scales=3):
        super().__init__()

        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(input_nc, ndf, n_layers)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(2, stride=2, padding=0)

    def forward(self, x):
        outputs = []
        for i, D in enumerate(self.discriminators):
            outputs.append(D(x))
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        return outputs


def lsgan_d_loss(real_preds, fake_preds):
    loss = 0
    for real_pred, fake_pred in zip(real_preds, fake_preds):
        loss += torch.mean((real_pred - 1) ** 2) + torch.mean(fake_pred ** 2)
    return loss * 0.5


def lsgan_g_loss(fake_preds):
    loss = 0
    for fake_pred in fake_preds:
        loss += torch.mean((fake_pred - 1) ** 2)
    return loss
