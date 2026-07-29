from re import A
from scipy.io import loadmat
import pydicom
import tifffile
import numpy as np
from models.KernelEstimator import KernelEstimator
import torch
import os
import matplotlib.pyplot as plt
from utils.utils import compute_psd, spline_to_kernel, generate_images

device = 'cuda'

def dicom_to_normalized(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    arr = arr * slope + intercept
    arr = np.clip(arr, -1000, 3000)
    arr = (arr + 1000) / 4000
    return arr

dcm_file_1 = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2030/I20"
ds_1 = pydicom.dcmread(dcm_file_1)
arr_1 = ds_1.pixel_array.astype(np.float32)
slope = float(getattr(ds_1, 'RescaleSlope', 1))
intercept = float(getattr(ds_1, 'RescaleIntercept', 0))
arr_1 = arr_1 * slope + intercept
arr_1 = np.clip(arr_1, -1000, 3000)
arr_1 = (arr_1 + 1000) / 4000

dcm_file_2 = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2030/I20"
ds_2 = pydicom.dcmread(dcm_file_2)
arr_2 = ds_2.pixel_array.astype(np.float32)
slope = float(getattr(ds_2, 'RescaleSlope', 1))
intercept = float(getattr(ds_2, 'RescaleIntercept', 0))
arr_2 = arr_2 * slope + intercept
arr_2 = np.clip(arr_2, -1000, 3000)
arr_2 = (arr_2 + 1000) / 4000

mask_paths = ["/home/cxv166/PhantomTesting/Code/masks/Mask_LD.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_RD.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_LU.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_RU.tif"]

mask_names = ['LD','RD','LU','RU']

masks = [tifffile.imread(p) for p in mask_paths]

# --- Ground truth (raw arr_1) results ---
gt_results = {}
for name, mask in zip(mask_names, masks):
    binary_mask = mask > 0

    if binary_mask.shape != arr_1.shape:
        raise ValueError(f"Shape mismatch, mask shape is {binary_mask.shape}, img shape is {arr_1.shape}")

    region_values = arr_1[binary_mask]
    mean_val = region_values.mean()
    std_val = region_values.std()
    mean_hu = mean_val * 4000 - 1000
    std_hu = std_val * 4000

    print(mean_hu, std_hu)
    gt_results[name] = [mean_hu, std_hu]

model = KernelEstimator()
model.to('cuda')
checkpoint = torch.load('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_kernel256/checkpoints/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

arr_1 = torch.from_numpy(arr_1)
arr_2 = torch.from_numpy(arr_2)

psd_1 = compute_psd(arr_1.unsqueeze(0).unsqueeze(0), device='cuda').to('cuda')
psd_2 = compute_psd(arr_2.unsqueeze(0).unsqueeze(0), device='cuda').to('cuda')

with torch.no_grad():
    mtf_1 = model(psd_1)
    mtf_2 = model(psd_2)

kernel_1, kernel_2 = spline_to_kernel(mtf_1, mtf_2)

filter1to2 = kernel_2 / (kernel_1 + 1e-10)
filter2to1 = kernel_1 / (kernel_2 + 1e-10)

Image_generated1, Image_generated2 = generate_images(arr_1, arr_2, filter1to2, filter2to1)
Image_generated1 = Image_generated1.squeeze(0)

# --- Model-generated image results ---
model_results = {}
for name, mask in zip(mask_names, masks):
    binary_mask = mask > 0
    binary_mask = torch.from_numpy(binary_mask)

    if binary_mask.shape != Image_generated1.shape:
        raise ValueError(f"Shape mismatch, mask shape is {binary_mask.shape}, img shape is {Image_generated1.shape}")

    region_values = Image_generated1[binary_mask]
    mean_val = region_values.mean()
    std_val = region_values.std()
    mean_hu = mean_val * 4000 - 1000
    std_hu = std_val * 4000

    print(mean_hu.item(), std_hu.item())
    model_results[name] = [mean_hu.item(), std_hu.item()]

mtf_c = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_C_MTF_Results_mat.mat')
mtf_d = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_D_MTF_Results_mat.mat')

results = mtf_c['results']
r = results[0, 0]
mtf_axis_c = r['mtfAxis'][0]
mtf_val_c = r['mtfVal'][0]

results = mtf_d['results']
r = results[0, 0]
mtf_axis_d = r['mtfAxis'][0]
mtf_val_d = r['mtfVal'][0]

os.makedirs('/home/cxv166/PhantomTesting/Code/S2030_recon', exist_ok=True)
os.makedirs('/home/cxv166/PhantomTesting/Code/S2050_recon', exist_ok=True)

ds_c = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2030/I20')
ds_d = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2050/I20')

pixel_spacing_mm_c = float(ds_c.PixelSpacing[0])
pixel_spacing_mm_d = float(ds_d.PixelSpacing[0])
assert pixel_spacing_mm_d == pixel_spacing_mm_c, "pixel spacing must be the same size"
pixel_spacing_mm = pixel_spacing_mm_c

image_c = torch.from_numpy(dicom_to_normalized(ds_c)).unsqueeze(0).unsqueeze(0).to(device)
image_d = torch.from_numpy(dicom_to_normalized(ds_d)).unsqueeze(0).unsqueeze(0).to(device)
nyquist = 0.5 / pixel_spacing_mm  # cycles/mm

mask_c = mtf_axis_c <= nyquist
mask_d = mtf_axis_d <= nyquist
actual_max_freq = float(mtf_axis_c[mask_c].max())

mtf_v_c = torch.from_numpy(mtf_val_c[mask_c]).float().unsqueeze(0).to(device)
mtf_v_d = torch.from_numpy(mtf_val_d[mask_d]).float().unsqueeze(0).to(device)

kernel_c, kernel_d = spline_to_kernel(mtf_v_c, mtf_v_d)

filterCtoD = kernel_d / (kernel_c + 1e-10)
filterDtoC = kernel_c / (kernel_d + 1e-10)

image_d_generated, image_c_generated = generate_images(image_c, image_d, filterCtoD, filterDtoC)

image_d_generated = image_d_generated.squeeze(0).squeeze(0)
image_c_generated = image_c_generated.squeeze(0).squeeze(0)

# --- MTF-reconstructed image results ---
mtf_reconstructed_results = {}
for name, mask in zip(mask_names, masks):
    binary_mask = mask > 0
    binary_mask = torch.from_numpy(binary_mask)

    if binary_mask.shape != image_d_generated.shape:
        raise ValueError(f"Shape mismatch, mask shape is {binary_mask.shape}, img shape is {image_d_generated.shape}")

    region_values = image_d_generated[binary_mask]
    mean_val = region_values.mean()
    std_val = region_values.std()
    mean_hu = mean_val * 4000 - 1000
    std_hu = std_val * 4000

    print(mean_hu.item(), std_hu.item())
    mtf_reconstructed_results[name] = [mean_hu.item(), std_hu.item()]

plot_dir = '/home/cxv166/PhantomTesting/Code/plots'
os.makedirs(plot_dir, exist_ok=True)

labels = ['Ground Truth', 'Model Generated', 'MTF Reconstructed']

for name in mask_names:
    means = [gt_results[name][0], model_results[name][0], mtf_reconstructed_results[name][0]]
    stds = [gt_results[name][1], model_results[name][1], mtf_reconstructed_results[name][1]]

    x = np.arange(len(labels))
    colors = ['#4C72B0', '#DD8452', '#55A868']

    # --- Mean HU plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, means, color=colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean HU')
    ax.set_title(f'Mask {name}: Mean HU Comparison')
    ax.axhline(0, color='black', linewidth=0.8)

    fig.tight_layout()
    mean_out_path = os.path.join(plot_dir, f'{name}_mean_comparison.png')
    fig.savefig(mean_out_path, dpi=150)
    plt.close(fig)
    print(f"Saved mean plot for mask {name} to {mean_out_path}")

    # --- Std HU plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, stds, color=colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Std HU')
    ax.set_title(f'Mask {name}: Std HU Comparison')
    ax.axhline(0, color='black', linewidth=0.8)

    fig.tight_layout()
    std_out_path = os.path.join(plot_dir, f'{name}_std_comparison.png')
    fig.savefig(std_out_path, dpi=150)
    plt.close(fig)
    print(f"Saved std plot for mask {name} to {std_out_path}")
