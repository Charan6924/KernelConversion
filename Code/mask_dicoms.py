from scipy.io import loadmat
import pydicom
import tifffile
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
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


def to_numpy(img):
    """Convert a torch tensor (any device) or numpy array into a 2D numpy array."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return np.squeeze(img)


def get_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def compute_region_metrics(image, gt_image, masks, mask_names, label):
    """Compute SSIM and PSNR for each masked region of `image` against the
    corresponding region of `gt_image`. Both SSIM and PSNR are computed on
    the [0, 1]-normalized image domain (same domain the images are loaded in)."""
    image_np = to_numpy(image).astype(np.float64)
    gt_np = to_numpy(gt_image).astype(np.float64)

    results = {}
    for name, mask in zip(mask_names, masks):
        binary_mask = mask > 0

        if binary_mask.shape != image_np.shape:
            raise ValueError(
                f"Shape mismatch, mask shape is {binary_mask.shape}, img shape is {image_np.shape}"
            )
        if binary_mask.shape != gt_np.shape:
            raise ValueError(
                f"Shape mismatch, mask shape is {binary_mask.shape}, gt shape is {gt_np.shape}"
            )

        y0, y1, x0, x1 = get_bbox(binary_mask)
        img_crop = image_np[y0:y1, x0:x1]
        gt_crop = gt_np[y0:y1, x0:x1]

        data_range = gt_crop.max() - gt_crop.min()
        if data_range == 0:
            data_range = 1.0

        ssim_val = ssim(gt_crop, img_crop, data_range=data_range)
        psnr_val = psnr(gt_crop, img_crop, data_range=data_range)

        print(f'[{label}] {name}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB')
        results[name] = [ssim_val, psnr_val]
    return results


dcm_file_1 = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2040/I20"
dcm_file_2 = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2050/I20"

ds_1, arr_1 = load_and_normalize_dicom(dcm_file_1)
ds_2, arr_2 = load_and_normalize_dicom(dcm_file_2)

mask_paths = ["/home/cxv166/PhantomTesting/Code/masks/Mask_Center.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_LD.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_RD.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_LU.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_RU.tif"]

mask_names = ['Center', 'LD', 'Air', 'LU', 'Phantom']

masks = [tifffile.imread(p) for p in mask_paths]

# arr_1 is treated as the ground-truth reference image that every method's
# output is compared against via SSIM/PSNR.
ground_truth_image = arr_1

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
Image_generated2 = Image_generated2.squeeze(0)

model_results = compute_region_metrics(Image_generated2, ground_truth_image, masks, mask_names, label='Model Generated')

mtf_c = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_YA_MTF_Results_mat.mat')
mtf_d = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_YB_MTF_Results_mat.mat')

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

ds_c, norm_c = load_and_normalize_dicom('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2040/I20')
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

mtf_reconstructed_results = compute_region_metrics(
    image_d_generated, ground_truth_image, masks, mask_names, label='MTF Reconstructed'
)

cut_reconstructed_path = '/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_cut_dicom/S65840_S2030_smooth_to_sharp/0000.dcm'
ds_cut, arr_cut = load_and_normalize_dicom(cut_reconstructed_path)
cut_reconstructed_results = compute_region_metrics(
    arr_cut, ground_truth_image, masks, mask_names, label='Cut Reconstructed'
)

pix2pix_reconstructed_path = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_pix2pix_dicom/S65840_S2030_sharp_to_smooth/0000.dcm"
ds_pix2pix, arr_pix2pix = load_and_normalize_dicom(pix2pix_reconstructed_path)
pix2pix_reconstructed_results = compute_region_metrics(
    arr_pix2pix, ground_truth_image, masks, mask_names, label='Pix2Pix Reconstructed'
)

cyclegan_reconstructed_path = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_cyclegan_dicom/S65840_S2030_smooth_to_sharp/0000.dcm"
ds_cyclegan, arr_cyclegan = load_and_normalize_dicom(cyclegan_reconstructed_path)
cyclegan_reconstructed_results = compute_region_metrics(
    arr_cyclegan, ground_truth_image, masks, mask_names, label='CycleGan Reconstructed'
)

plot_dir = '/home/cxv166/PhantomTesting/Code/plots'
os.makedirs(plot_dir, exist_ok=True)

# Note: "Original Image" vs itself is trivial (SSIM=1, PSNR=inf) so it is
# excluded from the comparison plots; the ground truth is the reference,
# not a method being scored.
result_sets = [
    ('MTF Reconstructed', mtf_reconstructed_results),
    ('Model Generated', model_results),
    ('Cut Reconstructed', cut_reconstructed_results),
    ('Pix2Pix Reconstructed', pix2pix_reconstructed_results),
    ('CycleGan Reconstructed', cyclegan_reconstructed_results),
]

method_labels = [name for name, _ in result_sets]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

for name in mask_names:
    ssim_vals = [results[name][0] for _, results in result_sets]
    psnr_vals = [results[name][1] for _, results in result_sets]

    x = np.arange(len(method_labels))

    # SSIM plot
    fig_ssim, ax_ssim = plt.subplots(figsize=(9, 5))
    ax_ssim.bar(x, ssim_vals, color=colors[:len(method_labels)], width=0.6)
    ax_ssim.set_xticks(x)
    ax_ssim.set_xticklabels(method_labels, rotation=20, ha='right')
    ax_ssim.set_ylabel('SSIM')
    ax_ssim.set_ylim(0, 1)
    ax_ssim.set_title(f'{name}: SSIM vs Ground Truth')
    fig_ssim.tight_layout()
    ssim_out_path = os.path.join(plot_dir, f'{name}_ssim_comparison.png')
    fig_ssim.savefig(ssim_out_path, dpi=150)
    plt.close(fig_ssim)
    print(f"Saved SSIM plot for mask {name} to {ssim_out_path}")

    # PSNR plot
    fig_psnr, ax_psnr = plt.subplots(figsize=(9, 5))
    ax_psnr.bar(x, psnr_vals, color=colors[:len(method_labels)], width=0.6)
    ax_psnr.set_xticks(x)
    ax_psnr.set_xticklabels(method_labels, rotation=20, ha='right')
    ax_psnr.set_ylabel('PSNR (dB)')
    ax_psnr.set_title(f'{name}: PSNR vs Ground Truth')
    fig_psnr.tight_layout()
    psnr_out_path = os.path.join(plot_dir, f'{name}_psnr_comparison.png')
    fig_psnr.savefig(psnr_out_path, dpi=150)
    plt.close(fig_psnr)
    print(f"Saved PSNR plot for mask {name} to {psnr_out_path}")
