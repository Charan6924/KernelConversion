import numpy as np
import nibabel as nib
import torch
import os
import matplotlib
from torch.utils.data import DataLoader
matplotlib.use('Agg')
from scipy.io import loadmat
import pydicom
from utils.utils import generate_images, spline_to_kernel
import torch.nn.functional as F
from scipy.interpolate import interp1d
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

'''
Reconstructing phantom volumes with ground truth phantom measurements
'''

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

def compute_metrics(original, reconstructed, data_range=0.6):
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

def plot_filter_profiles(filter1to2, filter2to1, save_path=None):
    """
    Plot the row-255 profile of each filter.
    filter1to2, filter2to1: tensors of shape [..., H, W] (e.g. [1,1,512,512])
    """
    f1 = filter1to2[:, 255, :]
    f2 = filter2to1[:, 255, :]

    if torch.is_tensor(f1):
        f1 = f1.squeeze().detach().cpu().numpy()
    if torch.is_tensor(f2):
        f2 = f2.squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f1, label='filter1to2 (row 255)', color='steelblue', linewidth=2)
    ax.plot(f2, label='filter2to1 (row 255)', color='tomato', linewidth=2)

    ax.set_xlabel('Pixel index', fontsize=12)
    ax.set_ylabel('Filter value', fontsize=12)
    ax.set_title('Filter Profiles at Row 255', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Filter profile plot saved to {save_path}")
    plt.show()

mtf_c = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_C_MTF_Results_mat.mat')
mtf_d = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_YA_MTF_Results_mat.mat')

ds_c = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2020/I20')  # C (sharp)
ds_d = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2040/I20')  # YB (smooth)

results = mtf_c['results']
r = results[0, 0]
mtf_axis_c = r['mtfAxis'][0]    # shape (64,)
mtf_val_c  = r['mtfVal'][0]     # shape (64,)

results = mtf_d['results']
r = results[0, 0]
mtf_axis_d = r['mtfAxis'][0]    # shape (64,)
mtf_val_d  = r['mtfVal'][0]

pixel_spacing_mm_c = float(ds_c.PixelSpacing[0])
pixel_spacing_mm_d = float(ds_d.PixelSpacing[0])
assert pixel_spacing_mm_d == pixel_spacing_mm_c, "pixel spacing must be the same size"

nyquist = 1 / (2 * pixel_spacing_mm_c)  # = 1.1132 lp/mm for this scanner
mask_c = mtf_axis_c <= nyquist
mtf_axis_c = mtf_axis_c[mask_c]
mtf_val_c  = mtf_val_c[mask_c]

mask_d = mtf_axis_d <= nyquist
mtf_axis_d = mtf_axis_d[mask_d]
mtf_val_d  = mtf_val_d[mask_d]

# Now convert to torch
mtf_val_c = torch.from_numpy(mtf_val_c).float().unsqueeze(0).to('cuda')
mtf_val_d = torch.from_numpy(mtf_val_d).float().unsqueeze(0).to('cuda')
mtf_axis_c = torch.from_numpy(mtf_axis_c)
mtf_axis_d = torch.from_numpy(mtf_axis_d)

kernel_c, kernel_d = spline_to_kernel(mtf_val_c, mtf_val_d)

lambda_for = 1e-10 * kernel_d.max().item()
lambda_rev = 1e-10 * kernel_c.max().item()

filterCtoD = (kernel_d * kernel_c/ (kernel_c**2 + lambda_rev)).to('cuda')
filterDtoC = (kernel_c * kernel_d/ (kernel_d**2 + lambda_for)).to('cuda')

plot_filter_profiles(filterCtoD,filterDtoC,save_path='/home/cxv166/PhantomTesting/Code/filter_profiles2.png')

image_d = torch.from_numpy(dicom_to_normalized(ds_d)).unsqueeze(0).unsqueeze(0).to('cuda')
image_c = torch.from_numpy(dicom_to_normalized(ds_c)).unsqueeze(0).unsqueeze(0).to('cuda')

# generate_images(I_smooth, I_sharp, filter_smooth2sharp, filter_sharp2smooth)
image_d_generated, image_c_generated = generate_images(image_c, image_d, filterCtoD, filterDtoC)

print("image_d range:", image_d.min().item(), image_d.max().item())
print("image_c range:", image_c.min().item(), image_c.max().item())
print("image_d_generated range:", image_d_generated.min().item(), image_d_generated.max().item())
print("image_c_generated range:", image_c_generated.min().item(), image_c_generated.max().item())

print("mean diff CtoD:", (image_d - image_d_generated).abs().mean().item())
print("mean diff DtoC:", (image_c - image_c_generated).abs().mean().item())

os.makedirs('/home/cxv166/PhantomTesting/Code/S2040_recon', exist_ok=True)
os.makedirs('/home/cxv166/PhantomTesting/Code/S2020_recon', exist_ok=True)

save_to_dicom(ds_d, image_d_generated, '/home/cxv166/PhantomTesting/Code/S2040_recon/I20')
save_to_dicom(ds_c, image_c_generated, '/home/cxv166/PhantomTesting/Code/S2020_recon/I20')

psnr_CtoD, ssim_CtoD = compute_metrics(image_d, image_d_generated)  # real D vs synthetic D
psnr_DtoC, ssim_DtoC = compute_metrics(image_c, image_c_generated)  # real C vs synthetic C

print(f"C → D | PSNR: {psnr_CtoD:.2f} dB  | SSIM: {ssim_CtoD:.4f}")
print(f"D → C | PSNR: {psnr_DtoC:.2f} dB  | SSIM: {ssim_DtoC:.4f}")
