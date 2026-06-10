'''using the trained model to reconstruct phantom volumes'''

import torch
import pydicom
import numpy as np
from models.KernelEstimator import KernelEstimator
import os
from utils.utils import compute_psd,compute_fft, generate_images, spline_to_kernel
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from data.pair_dicoms import KernelPairDataset
import os
import pydicom
from torch.utils.data import DataLoader
from pathlib import Path

model = KernelEstimator()
model.to('cuda')
checkpoint = torch.load('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_kernel256/checkpoints/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print('loaded model')

dataset = KernelPairDataset(root_dir = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840")
print(len(dataset))
loader = DataLoader(dataset)
print('loaded dataset')

def save_to_dicom(original_ds, reconstructed_array, output_path):
    ds = original_ds  
    arr = reconstructed_array.squeeze().cpu().numpy() 
    arr = (arr * 4000) - 1000  
    arr = np.clip(arr, -1000, 3000)
    arr = (arr - float(ds.RescaleIntercept)) / float(ds.RescaleSlope) 
    arr = np.round(arr).astype(np.uint16)
    
    ds.PixelData = arr.tobytes()
    ds.save_as(output_path)

def compute_metrics(original, reconstructed, data_range=1.0):
    """
    Compute PSNR and SSIM between original and reconstructed images.
    Inputs can be torch tensors [1,1,H,W] or numpy arrays.
    """
    if torch.is_tensor(original):
        original = original.squeeze().cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.squeeze().cpu().numpy()

    psnr_val = psnr(original, reconstructed, data_range=data_range)
    ssim_val = ssim(original, reconstructed, data_range=data_range)

    return psnr_val, ssim_val

psnr_sm2sh_list = []
ssim_sm2sh_list = []
psnr_sh2sm_list = []
ssim_sh2sm_list = []

for d in loader:
    smooth_vol = d['smooth'].squeeze(0)
    sharp_vol = d['sharp'].squeeze(0)
    print(smooth_vol.shape)

    assert sharp_vol.shape[0] == smooth_vol.shape[0], "both sharp and smooth volumes should have the same number of slices"

    for k in range(smooth_vol.shape[0]):
        smooth_slice = smooth_vol[k]
        sharp_slice = sharp_vol[k]

        smooth_slice_clipped = smooth_slice.clip(-1000, 3000)
        smooth_slice_normalized = (smooth_slice_clipped + 1000) / 4000

        sharp_slice_clipped = sharp_slice.clip(-1000, 3000)
        sharp_slice_normalized = (sharp_slice_clipped + 1000) / 4000

        smooth_psd = compute_psd(smooth_slice_normalized.unsqueeze(0).unsqueeze(0), device='cuda')
        sharp_psd = compute_psd(sharp_slice_normalized.unsqueeze(0).unsqueeze(0), device='cuda')

        smooth_psd = smooth_psd.to('cuda')
        sharp_psd = sharp_psd.to('cuda')
        print(smooth_psd.shape)

        with torch.no_grad():
            smooth_mtf = model(smooth_psd)
            sharp_mtf = model(sharp_psd)

        smooth_kernel, sharp_kernel = spline_to_kernel(smooth_mtf, sharp_mtf)

        sm2sh = sharp_kernel / (smooth_kernel + 1e-10)
        sh2sm = smooth_kernel / (sharp_kernel + 1e-10)

        I_generated_smooth, I_generated_sharp = generate_images(smooth_slice_normalized, sharp_slice_normalized, sm2sh, sh2sm)

        I_generated_smooth = (I_generated_smooth * 4000) - 1000
        I_generated_smooth = I_generated_smooth.clip(-1000, 3000)

        I_generated_sharp = (I_generated_sharp * 4000) - 1000
        I_generated_sharp = I_generated_sharp.clip(-1000, 3000)

        psnr_sm2sh, ssim_sm2sh = compute_metrics(sharp_slice_clipped, I_generated_sharp, data_range=4000)
        psnr_sh2sm, ssim_sh2sm = compute_metrics(smooth_slice_clipped, I_generated_smooth, data_range=4000)

        psnr_sm2sh_list.append(psnr_sm2sh)
        ssim_sm2sh_list.append(ssim_sm2sh)
        psnr_sh2sm_list.append(psnr_sh2sm)
        ssim_sh2sm_list.append(ssim_sh2sm)

print(f"smooth→sharp | PSNR: {np.mean(psnr_sm2sh_list):.2f} dB  SSIM: {np.mean(ssim_sm2sh_list):.4f}")
print(f"sharp→smooth | PSNR: {np.mean(psnr_sh2sm_list):.2f} dB  SSIM: {np.mean(ssim_sh2sm_list):.4f}")

# --- Metric plots ---
slices = range(1, len(psnr_sm2sh_list) + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Per-Slice Metrics", fontsize=14)

# PSNR: smooth→sharp
axes[0, 0].plot(slices, psnr_sm2sh_list, color='steelblue', linewidth=1)
axes[0, 0].axhline(np.mean(psnr_sm2sh_list), color='steelblue', linestyle='--', linewidth=1.2, label=f"Mean: {np.mean(psnr_sm2sh_list):.2f} dB")
axes[0, 0].set_title("PSNR — smooth → sharp")
axes[0, 0].set_xlabel("Slice index")
axes[0, 0].set_ylabel("PSNR (dB)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# PSNR: sharp→smooth
axes[0, 1].plot(slices, psnr_sh2sm_list, color='darkorange', linewidth=1)
axes[0, 1].axhline(np.mean(psnr_sh2sm_list), color='darkorange', linestyle='--', linewidth=1.2, label=f"Mean: {np.mean(psnr_sh2sm_list):.2f} dB")
axes[0, 1].set_title("PSNR — sharp → smooth")
axes[0, 1].set_xlabel("Slice index")
axes[0, 1].set_ylabel("PSNR (dB)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# SSIM: smooth→sharp
axes[1, 0].plot(slices, ssim_sm2sh_list, color='steelblue', linewidth=1)
axes[1, 0].axhline(np.mean(ssim_sm2sh_list), color='steelblue', linestyle='--', linewidth=1.2, label=f"Mean: {np.mean(ssim_sm2sh_list):.4f}")
axes[1, 0].set_title("SSIM — smooth → sharp")
axes[1, 0].set_xlabel("Slice index")
axes[1, 0].set_ylabel("SSIM")
axes[1, 0].set_ylim(0, 1)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# SSIM: sharp→smooth
axes[1, 1].plot(slices, ssim_sh2sm_list, color='darkorange', linewidth=1)
axes[1, 1].axhline(np.mean(ssim_sh2sm_list), color='darkorange', linestyle='--', linewidth=1.2, label=f"Mean: {np.mean(ssim_sh2sm_list):.4f}")
axes[1, 1].set_title("SSIM — sharp → smooth")
axes[1, 1].set_xlabel("Slice index")
axes[1, 1].set_ylabel("SSIM")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("metrics_per_slice.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved metrics_per_slice.png")
