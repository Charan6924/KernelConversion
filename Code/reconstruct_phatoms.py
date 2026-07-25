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
device = "cuda"

def torch_interp1d(x, y, x_new, fill_value=0.0):
    """
    1D linear interpolation mirroring MATLAB's interp1(x, y, x_new, 'linear', fill_value).
    x must be sorted ascending, 1D. x_new can be any shape.
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    orig_shape = x_new.shape
    xq = x_new.reshape(-1)

    # searchsorted gives insertion index; clamp so x0/x1 are always valid neighbors
    idx = torch.searchsorted(x, xq)
    idx = torch.clamp(idx, 1, x.shape[0] - 1)

    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]

    slope = (y1 - y0) / (x1 - x0)
    yq = y0 + slope * (xq - x0)

    # extrapolate to fill_value outside [x.min(), x.max()], matching MATLAB's interp1(...,0)
    out_of_range = (xq < x[0]) | (xq > x[-1])
    yq = torch.where(out_of_range, torch.full_like(yq, fill_value), yq)

    return yq.reshape(orig_shape)


def radial_to_2d(mtf_axis, mtf_val, grid_size=512, pixel_spacing_mm=0.44921875):
    """
    Direct port of mtf_2D.m: interpolate a 1D MTF curve (in cycles/mm) onto a
    2D radial frequency grid (in cycles/mm), DC at center, zero-filled beyond
    the input axis range.

    mtf_axis, mtf_val: 1D tensors (cycles/mm, MTF value), same length, axis ascending.
    Returns: (grid_size, grid_size) tensor.
    """
    device = mtf_val.device
    N = grid_size
    pix_size = pixel_spacing_mm

    # matches MATLAB: fx_mm = ((-floor(N/2):ceil(N/2)-1)/N)/pixSize
    idx = torch.arange(-(N // 2), N - N // 2, device=device, dtype=torch.float32)
    fx_mm = (idx / N) / pix_size

    FY, FX = torch.meshgrid(fx_mm, fx_mm, indexing='ij')
    R = torch.sqrt(FX**2 + FY**2)

    mtf_axis = mtf_axis.to(device=device, dtype=torch.float32)
    mtf_val = mtf_val.to(device=device, dtype=torch.float32)

    mtf2d = torch_interp1d(mtf_axis, mtf_val, R, fill_value=0.0)
    return mtf2d


def spline_to_kernel(mtf_axis_c, mtf_val_c, mtf_axis_d, mtf_val_d,
                      grid_size=512, pixel_spacing_mm=0.44921875):
    """
    Each curve now uses its own trimmed frequency axis, exactly like the two
    separate mtf_2D(...) calls in MATLAB — no shared/assumed max frequency.
    """
    kernel_c = radial_to_2d(mtf_axis_c, mtf_val_c, grid_size, pixel_spacing_mm).clamp(min=1e-6)
    kernel_d = radial_to_2d(mtf_axis_d, mtf_val_d, grid_size, pixel_spacing_mm).clamp(min=1e-6)
    return kernel_c, kernel_d


def regularized_filter(numerator_kernel, denominator_kernel, max_gain=5.0, lambda_frac=1e-3):
    """
    Wiener-style regularized ratio filter, mirroring the MATLAB:
        H = (A .* B) ./ (B.^2 + lambda);
        H = min(H, maxGain);

    numerator_kernel / denominator_kernel roles follow the same convention as
    the MATLAB script's H_D_to_C / H_C_to_D construction: the *target* MTF is
    numerator_kernel, and denominator_kernel is the system being deconvolved.
    """
    lam = lambda_frac * (denominator_kernel.max() ** 2)
    H = (numerator_kernel * denominator_kernel) / (denominator_kernel**2 + lam)
    H = torch.clamp(H, max=max_gain)
    return H


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
    f1 = filter1to2[:, 255]
    f2 = filter2to1[:, 255]
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
pixel_spacing_mm = pixel_spacing_mm_c

image_c = torch.from_numpy(dicom_to_normalized(ds_c)).unsqueeze(0).unsqueeze(0).to(device)
image_d = torch.from_numpy(dicom_to_normalized(ds_d)).unsqueeze(0).unsqueeze(0).to(device)
nyquist = 0.5 / pixel_spacing_mm  # cycles/mm

mask_c = mtf_axis_c <= nyquist
mask_d = mtf_axis_d <= nyquist
actual_max_freq = float(mtf_axis_c[mask_c].max())

mtf_v_c = torch.from_numpy(mtf_val_c[mask_c]).float().unsqueeze(0).to(device)
mtf_v_d = torch.from_numpy(mtf_val_d[mask_d]).float().unsqueeze(0).to(device)

# ---------------------------------------------------------------------------
# Build 2D MTF kernels (shared for both filter directions -- same physical
# pixel spacing and same truncated MTF curves feed both)
# ---------------------------------------------------------------------------
kernel_c, kernel_d = spline_to_kernel(
    mtf_axis_c[mask_c].clone().detach() if torch.is_tensor(mtf_axis_c) else torch.from_numpy(mtf_axis_c[mask_c]).float(),
    mtf_v_c,
    torch.from_numpy(mtf_axis_d[mask_d]).float() if not torch.is_tensor(mtf_axis_d) else mtf_axis_d[mask_d],
    mtf_v_d,
    pixel_spacing_mm=pixel_spacing_mm,
)
# ---------------------------------------------------------------------------
# Regularized filters (Wiener-style, matching the MATLAB H_D_to_C / H_C_to_D
# construction, with a gain ceiling to suppress high-frequency blow-up)
# FIX: previously a plain ratio with only a 1e-10 epsilon and no gain clamp,
# which is unstable near/above Nyquist where MTF -> 0.
# ---------------------------------------------------------------------------
filterCtoD = regularized_filter(kernel_d, kernel_c, max_gain=5.0)
filterDtoC = regularized_filter(kernel_c, kernel_d, max_gain=5.0)

plot_filter_profiles(filterCtoD, filterDtoC,
                     save_path='/home/cxv166/PhantomTesting/Code/filter_profiles2.png')

image_d_generated, image_c_generated = generate_images(image_c, image_d, filterCtoD, filterDtoC)

os.makedirs('/home/cxv166/PhantomTesting/Code/S2030_recon', exist_ok=True)
os.makedirs('/home/cxv166/PhantomTesting/Code/S2020_recon', exist_ok=True)
save_to_dicom(ds_d, image_d_generated, '/home/cxv166/PhantomTesting/Code/S2030_recon/I20')
save_to_dicom(ds_c, image_c_generated, '/home/cxv166/PhantomTesting/Code/S2020_recon/I20')

psnr_CtoD, ssim_CtoD = compute_metrics(image_d, image_d_generated)
psnr_DtoC, ssim_DtoC = compute_metrics(image_c, image_c_generated)
print(f"C -> D | PSNR: {psnr_CtoD:.2f} dB  | SSIM: {ssim_CtoD:.4f}")
print(f"D -> C | PSNR: {psnr_DtoC:.2f} dB  | SSIM: {ssim_DtoC:.4f}")
