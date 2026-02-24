import torch 
from PSDDataset import PSDDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def compute_psd_batch_gpu(image_batch, device):
    if image_batch.device.type != device:
        image_batch = image_batch.to(device)
    freq_map = torch.fft.fftshift(torch.fft.fft2(image_batch), dim=(-2, -1))
    psd = torch.abs(freq_map) ** 2
    psd = torch.log(psd + 1)
    b, c, h, w = psd.shape
    psd_flat = psd.view(b, -1)
    p_min = psd_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    p_max = psd_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    psd = (psd - p_min) / (p_max - p_min + 1e-8)
    return psd

def symmetric_average_2d(psd_2d):
    B, C, H, W = psd_2d.shape
    center = W // 2

    left  = psd_2d[:, :, :, :center].flip(dims=[-1])   # [B, 1, H, 256]
    right = psd_2d[:, :, :, center+1:]                  # [B, 1, H, 255]

    min_len = min(left.shape[-1], right.shape[-1])
    left  = left[:, :, :, :min_len]
    right = right[:, :, :, :min_len]

    averaged = (left + right) / 2.0                     # [B, 1, H, 255]

    dc = psd_2d[:, :, :, center].unsqueeze(-1)          # [B, 1, H, 1]

    smoothed = torch.cat([averaged.flip(dims=[-1]), dc, averaged], dim=-1)  # [B, 1, H, 511]
    return smoothed

image_dataset = PSDDataset(root_dir=r"D:\Charan work file\KernelEstimator\Data_Root")
image_train_loader = DataLoader(image_dataset, batch_size=16, shuffle=False,
                                num_workers=0, pin_memory=True)

I_smooth,I_sharp,_,_ = next(iter(image_train_loader))

def gaussian_blur_2d(x, kernel_size=21, sigma=5.0):
    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32) - pad
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0).to(x.device)
    kernel = kernel.expand(x.shape[1], 1, -1, -1)
    return F.conv2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'), 
                    kernel, groups=x.shape[1])

psd_smooth = compute_psd_batch_gpu(I_smooth, device='cuda')
psd_sharp  = compute_psd_batch_gpu(I_sharp,  device='cuda')
sym_smooth = symmetric_average_2d(psd_smooth)
sym_sharp  = symmetric_average_2d(psd_sharp)
blur_smooth = gaussian_blur_2d(sym_smooth)
blur_sharp  = gaussian_blur_2d(sym_sharp)
ratio = (blur_sharp / (blur_smooth + 1)).clip(0, 8)
ratio /= 8
center_row = ratio[0, 0, 0, :]
plt.plot(center_row.to('cpu'))
plt.show()