import numpy as np
import nibabel as nib
import torch
import os
import matplotlib
matplotlib.use('Agg')
from scipy.io import loadmat
import pydicom
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from utils.utils import generate_images

'''
Reconstructing phantom volumes with ground truth phantom measurements
'''

def radial_to_2d(radial_profile, grid_size=512, pixel_spacing_mm=0.44921875, mtf_max_freq=None):
    batch_size = radial_profile.shape[0]
    device = radial_profile.device
    n_points = radial_profile.shape[-1]
    center = grid_size / 2.0

    y = torch.arange(grid_size, device=device, dtype=torch.float32) - center
    x = torch.arange(grid_size, device=device, dtype=torch.float32) - center
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    nyquist = 1 / (2 * pixel_spacing_mm)  # ~1.113 lp/mm

    if mtf_max_freq is None:
        mtf_max_freq = nyquist

    distance = torch.sqrt(x_grid**2 + y_grid**2)
    freq = (distance / center) * nyquist
    t = freq / mtf_max_freq
    t = torch.clamp(t, 0, 1)

    profile = radial_profile.view(batch_size, 1, 1, n_points)
    grid_x = 2.0 * t - 1.0
    grid_y = torch.zeros_like(grid_x)
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    sampling_grid = sampling_grid.expand(batch_size, -1, -1, -1)

    kernel_2d = F.grid_sample(
        profile, sampling_grid,
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(1)  # (B, grid_size, grid_size)

    return kernel_2d


def spline_to_kernel(smooth_curve, sharp_curve, grid_size=512, mtf_max_freq=None):
    otf_smooth = radial_to_2d(smooth_curve, grid_size, mtf_max_freq=mtf_max_freq).clamp(min=1e-6)
    otf_sharp  = radial_to_2d(sharp_curve,  grid_size, mtf_max_freq=mtf_max_freq).clamp(min=1e-6)
    return otf_smooth, otf_sharp


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
    if torch.is_tensor(original):
        original = original.squeeze().cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.squeeze().cpu().numpy()
    psnr_val = psnr(original, reconstructed, data_range=data_range)
    ssim_val = ssim(original, reconstructed, data_range=data_range)
    return psnr_val, ssim_val


def plot_filter_profiles(filter1to2, filter2to1, save_path=None):
    f1 = filter1to2[:, 255, :]
    f2 = filter2to1[:, 255, :]
    if torch.is_tensor(f1):
        f1 = f1.squeeze().detach().cpu().numpy()
    if torch.is_tensor(f2):
        f2 = f2.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f1, label='filterCtoD (row 255)', color='steelblue', linewidth=2)
    ax.plot(f2, label='filterDtoC (row 255)', color='tomato', linewidth=2)
    ax.set_xlabel('Pixel index', fontsize=12)
    ax.set_ylabel('Filter value', fontsize=12)
    ax.set_title('Filter Profiles at Row 255', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Filter profile plot saved to {save_path}")

mtf_c = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_C_MTF_Results_mat.mat')
mtf_d = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_D_MTF_Results_mat.mat')

results = mtf_c['results']
r = results[0, 0]
mtf_axis_c = r['mtfAxis'][0]
mtf_val_c  = r['mtfVal'][0]

results = mtf_d['results']
r = results[0, 0]
mtf_axis_d = r['mtfAxis'][0]
mtf_val_d  = r['mtfVal'][0]

ds_c = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2020/I20')  # C
ds_d = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2030/I20')  # D

pixel_spacing_mm_c = float(ds_c.PixelSpacing[0])
pixel_spacing_mm_d = float(ds_d.PixelSpacing[0])
assert pixel_spacing_mm_d == pixel_spacing_mm_c, "pixel spacing must be the same size"

image_c = torch.from_numpy(dicom_to_normalized(ds_c)).unsqueeze(0).unsqueeze(0).to('cuda')
image_d = torch.from_numpy(dicom_to_normalized(ds_d)).unsqueeze(0).unsqueeze(0).to('cuda')

mask_c_CtD = mtf_axis_c <= 1.167
mask_d_CtD = mtf_axis_d <= 1.167
actual_max_freq_CtD = float(mtf_axis_c[mask_c_CtD].max())
mtf_v_c_CtD = torch.from_numpy(mtf_val_c[mask_c_CtD]).float().unsqueeze(0).to('cuda')
mtf_v_d_CtD = torch.from_numpy(mtf_val_d[mask_d_CtD]).float().unsqueeze(0).to('cuda')
kernel_c_CtD, kernel_d_CtD = spline_to_kernel(mtf_v_c_CtD, mtf_v_d_CtD, mtf_max_freq=actual_max_freq_CtD)
filterCtoD = (kernel_d_CtD / (kernel_c_CtD + 1e-10)).to('cuda')

mask_c_DtC = mtf_axis_c <= 1.256
mask_d_DtC = mtf_axis_d <= 1.256
actual_max_freq_DtC = float(mtf_axis_c[mask_c_DtC].max())
mtf_v_c_DtC = torch.from_numpy(mtf_val_c[mask_c_DtC]).float().unsqueeze(0).to('cuda')
mtf_v_d_DtC = torch.from_numpy(mtf_val_d[mask_d_DtC]).float().unsqueeze(0).to('cuda')
kernel_c_DtC, kernel_d_DtC = spline_to_kernel(mtf_v_c_DtC, mtf_v_d_DtC, mtf_max_freq=actual_max_freq_DtC)
filterDtoC = (kernel_c_DtC / (kernel_d_DtC + 1e-10)).to('cuda')

plot_filter_profiles(filterCtoD, filterDtoC,
                     save_path='/home/cxv166/PhantomTesting/Code/filter_profiles2.png')

image_d_generated, image_c_generated = generate_images(image_c, image_d, filterCtoD, filterDtoC)

print("image_d range:", image_d.min().item(), image_d.max().item())
print("image_c range:", image_c.min().item(), image_c.max().item())
print("image_d_generated range:", image_d_generated.min().item(), image_d_generated.max().item())
print("image_c_generated range:", image_c_generated.min().item(), image_c_generated.max().item())
print("mean diff CtoD:", (image_d - image_d_generated).abs().mean().item())
print("mean diff DtoC:", (image_c - image_c_generated).abs().mean().item())

os.makedirs('/home/cxv166/PhantomTesting/Code/S2030_recon', exist_ok=True)
os.makedirs('/home/cxv166/PhantomTesting/Code/S2020_recon', exist_ok=True)
save_to_dicom(ds_d, image_d_generated, '/home/cxv166/PhantomTesting/Code/S2030_recon/I20')
save_to_dicom(ds_c, image_c_generated, '/home/cxv166/PhantomTesting/Code/S2020_recon/I20')

psnr_CtoD, ssim_CtoD = compute_metrics(image_d, image_d_generated)
psnr_DtoC, ssim_DtoC = compute_metrics(image_c, image_c_generated)
print(f"C → D | PSNR: {psnr_CtoD:.2f} dB  | SSIM: {ssim_CtoD:.4f}")
print(f"D → C | PSNR: {psnr_DtoC:.2f} dB  | SSIM: {ssim_DtoC:.4f}")


'''(phantomtesting) [cxv166@gput074 Code]$ python reconstruct_phatoms.py
Filter profile plot saved to /home/cxv166/PhantomTesting/Code/filter_profiles2.png
image_d range: 0.0 0.5504999756813049
image_c range: 0.0 0.49050000309944153
image_d_generated range: 0.0 0.5008903741836548
image_c_generated range: 0.0 0.5358105897903442
mean diff CtoD: 0.005078439600765705
mean diff DtoC: 0.004609832540154457
C → D | PSNR: 36.60 dB  | SSIM: 0.9568
D → C | PSNR: 37.19 dB  | SSIM: 0.9726'''
