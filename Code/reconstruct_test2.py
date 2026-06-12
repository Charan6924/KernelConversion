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

def dicom_to_normalized(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    arr = arr * slope + intercept 
    arr = np.clip(arr, -1000, 3000)
    arr = (arr + 1000) / 4000      
    return arr

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
    print(f"[DEBUG] original    min={original.min():.4f}, max={original.max():.4f}")
    print(f"[DEBUG] reconstructed min={reconstructed.min():.4f}, max={reconstructed.max():.4f}")
    print(f"[DEBUG] data_range={data_range}")

    psnr_val = psnr(original, reconstructed, data_range=data_range)
    ssim_val = ssim(original, reconstructed, data_range=data_range)

    return psnr_val, ssim_val

def plot_predicted_mtfs(mtf_1, mtf_2, save_path=None):
    """
    Plot the two predicted MTF curves from the KernelEstimator model.
    mtf_1, mtf_2: model outputs, shape [1, N] or [N]
    """
    m1 = mtf_1.squeeze().cpu().numpy()
    m2 = mtf_2.squeeze().cpu().numpy()

    freqs = np.linspace(0, 1, len(m1))  # Normalized spatial frequency [0, Nyquist]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, m1, label='S2030 (smooth kernel)', color='steelblue', linewidth=2)
    ax.plot(freqs, m2, label='S2050 (sharp kernel)', color='tomato', linewidth=2)

    ax.set_xlabel('Normalized Spatial Frequency (cycles/pixel)', fontsize=12)
    ax.set_ylabel('MTF', fontsize=12)
    ax.set_title('Predicted MTF — KernelEstimator', fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='MTF50')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"MTF plot saved to {save_path}")
    plt.show()

ds1 = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2030/I20')
pixel_array1 = ds1.pixel_array
ds2 = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2050/I20')
pixel_array2 = ds2.pixel_array

model = KernelEstimator()
model.to('cuda')
checkpoint = torch.load('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_kernel256/checkpoints/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

pixel_array1 = torch.from_numpy(dicom_to_normalized(ds1)).unsqueeze(0).unsqueeze(0)
pixel_array2 = torch.from_numpy(dicom_to_normalized(ds2)).unsqueeze(0).unsqueeze(0)

psd_1 = compute_psd(pixel_array1, device = 'cuda').to('cuda')
psd_2 = compute_psd(pixel_array2, device = 'cuda').to('cuda')

print(psd_1.shape)

with torch.no_grad():
    mtf_1 = model(psd_1)
    mtf_2 = model(psd_2)
    plot_predicted_mtfs(mtf_1, mtf_2, save_path='/home/cxv166/PhantomTesting/Code/mtf_comparison.png')

kernel_1, kernel_2 = spline_to_kernel(mtf_1,mtf_2)

filter1to2 = kernel_2/(kernel_1 + 1e-10)
filter2to1 = kernel_1/(kernel_2 + 1e-10)

Image_generated1, Image_generated2 = generate_images(pixel_array1,pixel_array2,filter1to2,filter2to1)

os.makedirs('/home/cxv166/PhantomTesting/Code/S2030_recon', exist_ok=True)
os.makedirs('/home/cxv166/PhantomTesting/Code/S2050_recon', exist_ok=True)

save_to_dicom(ds1, Image_generated1, '/home/cxv166/PhantomTesting/Code/S2030_recon/I20')
save_to_dicom(ds2, Image_generated2, '/home/cxv166/PhantomTesting/Code/S2050_recon/I20')

psnr_1to2, ssim_1to2 = compute_metrics(pixel_array2, Image_generated2)
psnr_2to1, ssim_2to1 = compute_metrics(pixel_array1, Image_generated1)

print("=== Reconstruction Metrics ===")
print(f"S2030 → S2050 | PSNR: {psnr_1to2:.2f} dB  | SSIM: {ssim_1to2:.4f}")
print(f"S2050 → S2030 | PSNR: {psnr_2to1:.2f} dB  | SSIM: {ssim_2to1:.4f}")
