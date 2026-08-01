from scipy.io import loadmat
import pydicom
import tifffile
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from models.KernelEstimator import KernelEstimator
from utils.utils import compute_psd, spline_to_kernel, generate_images

device = 'cuda'


def load_and_normalize_dicom(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    arr = arr * slope + intercept
    arr = np.clip(arr, -1000, 3000)
    arr = (arr + 1000) / 4000
    return ds, arr


def denormalize_to_hu(region_values):
    """Undo the [0, 1] normalization back to HU units (mean, std)."""
    mean_val = region_values.mean()
    std_val = region_values.std()
    mean_hu = mean_val * 4000 - 1000
    std_hu = std_val * 4000
    return mean_hu, std_hu


def compute_region_stats(image, masks, mask_names, label):
    results = {}
    for name, mask in zip(mask_names, masks):
        binary_mask = mask > 0

        if isinstance(image, torch.Tensor):
            binary_mask_t = torch.from_numpy(binary_mask)
            if binary_mask_t.shape != image.shape:
                raise ValueError(
                    f"Shape mismatch, mask shape is {binary_mask_t.shape}, img shape is {image.shape}"
                )
            region_values = image[binary_mask_t]
            mean_hu, std_hu = denormalize_to_hu(region_values)
            mean_hu, std_hu = mean_hu.item(), std_hu.item()
        else:
            if binary_mask.shape != image.shape:
                raise ValueError(
                    f"Shape mismatch, mask shape is {binary_mask.shape}, img shape is {image.shape}"
                )
            region_values = image[binary_mask]
            mean_hu, std_hu = denormalize_to_hu(region_values)

        print(f'[{label}] {name}: mean={mean_hu:.2f} HU, std={std_hu:.2f} HU')
        results[name] = [mean_hu, std_hu]
    return results


dcm_file_1 = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2030/I20"
dcm_file_2 = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2050/I20"

ds_1, arr_1 = load_and_normalize_dicom(dcm_file_1)
ds_2, arr_2 = load_and_normalize_dicom(dcm_file_2)

mask_paths = ["/home/cxv166/PhantomTesting/Code/masks/Mask_LD.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_RD.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_LU.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_RU.tif"]

mask_names = ['LD', 'RD', 'LU', 'RU']

masks = [tifffile.imread(p) for p in mask_paths]

gt_results = compute_region_stats(arr_1, masks, mask_names, label='Ground Truth')

model = KernelEstimator()
model.to('cuda')
checkpoint = torch.load('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_kernel256/checkpoints/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

arr_1_t = torch.from_numpy(arr_1)
arr_2_t = torch.from_numpy(arr_2)

psd_1 = compute_psd(arr_1_t.unsqueeze(0).unsqueeze(0), device='cuda').to('cuda')
psd_2 = compute_psd(arr_2_t.unsqueeze(0).unsqueeze(0), device='cuda').to('cuda')

with torch.no_grad():
    mtf_1 = model(psd_1)
    mtf_2 = model(psd_2)

kernel_1, kernel_2 = spline_to_kernel(mtf_1, mtf_2)

filter1to2 = kernel_2 / (kernel_1 + 1e-10)
filter2to1 = kernel_1 / (kernel_2 + 1e-10)

Image_generated1, Image_generated2 = generate_images(arr_1_t, arr_2_t, filter1to2, filter2to1)
Image_generated1 = Image_generated1.squeeze(0)

model_results = compute_region_stats(Image_generated1, masks, mask_names, label='Model Generated')

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

ds_c, norm_c = load_and_normalize_dicom('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2030/I20')
ds_d, norm_d = load_and_normalize_dicom('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2050/I20')

pixel_spacing_mm_c = float(ds_c.PixelSpacing[0])
pixel_spacing_mm_d = float(ds_d.PixelSpacing[0])
assert pixel_spacing_mm_d == pixel_spacing_mm_c, "pixel spacing must be the same size"
pixel_spacing_mm = pixel_spacing_mm_c

image_c = torch.from_numpy(norm_c).unsqueeze(0).unsqueeze(0).to(device)
image_d = torch.from_numpy(norm_d).unsqueeze(0).unsqueeze(0).to(device)
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

mtf_reconstructed_results = compute_region_stats(
    image_d_generated, masks, mask_names, label='MTF Reconstructed'
)

cut_reconstructed_path = '/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_cut_dicom/S65840_S2050_smooth_to_sharp/0000.dcm'
ds_cut, arr_cut = load_and_normalize_dicom(cut_reconstructed_path)
cut_reconstructed_results = compute_region_stats(
    arr_cut, masks, mask_names, label='Cut Reconstructed'
)

pix2pix_reconstructed_path = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_pix2pix_dicom/S65840_S2030_sharp_to_smooth/0000.dcm"
ds_pix2pix, arr_pix2pix = load_and_normalize_dicom(pix2pix_reconstructed_path)
pix2pix_reconstructed_results = compute_region_stats(
    arr_pix2pix, masks, mask_names, label='Pix2Pix Reconstructed'
)

cyclegan_reconstructed_path = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_cyclegan_dicom/S65840_S2050_smooth_to_sharp/0000.dcm"
ds_cyclegan, arr_cyclegan = load_and_normalize_dicom(cyclegan_reconstructed_path)
cyclegan_reconstructed_results = compute_region_stats(
    arr_cyclegan, masks, mask_names, label='CycleGan Reconstructed'
)

plot_dir = '/home/cxv166/PhantomTesting/Code/plots'
os.makedirs(plot_dir, exist_ok=True)

result_sets = [
    ('Ground Truth', gt_results),
    ('Model Generated', model_results),
    ('MTF Reconstructed', mtf_reconstructed_results),
    # ('Cut Reconstructed', cut_reconstructed_results),
    ('Pix2Pix Reconstructed', pix2pix_reconstructed_results),
    ('CycleGan Reconstructed', cyclegan_reconstructed_results),
]

method_labels = [name for name, _ in result_sets]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

n_methods = len(result_sets)
n_masks = len(mask_names)

group_x = np.arange(n_masks)          # one position per mask (group center)
bar_width = 0.8 / n_methods           # width of each individual bar within a group

fig, ax = plt.subplots(figsize=(12, 6))

for i, (method_name, results) in enumerate(result_sets):
    means = [results[name][0] for name in mask_names]
    stds = [results[name][1] for name in mask_names]

    # offset each method's bars so they sit side-by-side within each mask's group
    offset = (i - (n_methods - 1) / 2) * bar_width
    ax.bar(group_x + offset, means, width=bar_width, yerr=stds, capsize=4,
           color=colors[i], label=method_name,
           error_kw={'elinewidth': 1.2, 'ecolor': 'black'})

ax.set_xticks(group_x)
ax.set_xticklabels(mask_names)
ax.set_xlabel('Mask')
ax.set_ylabel('Mean HU (\u00b1 std)')
ax.set_title('Mean HU with Std Error, by Mask and Method')
ax.axhline(0, color='black', linewidth=0.8)
ax.legend(loc='upper right', fontsize=9)

fig.tight_layout()
out_path = os.path.join(plot_dir, 'all_masks_mean_std_comparison.png')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved combined mean+std plot for all masks to {out_path}")
