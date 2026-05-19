import numpy as np
import nibabel as nib
import torch
import os
from torch.utils.data import DataLoader
from utils import generate_images, spline_to_kernel 
from TestDataset import TestDataset
from ddpTrainComplexKernel import complex_kernel_ratio, apply_complex_filter
from KernelEstimator import KernelEstimator
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KernelEstimator()
checkpoint = torch.load("/home/cxv166/PhantomTesting/Code/training_output_0.5/checkpoints/best_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() 
print('Loaded model successfully')

data_root = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
dataset = TestDataset(root_dir=data_root, preload=True)
print('Loaded test dataset')

output_dir = "/home/cxv166/PhantomTesting/reconstructions"
os.makedirs(output_dir, exist_ok=True)

def model_to_complex_kernel(out):
    out = out.float()
    output_size = 512
    device = 'cuda'
    dtype = out.dtype
    center = output_size // 2
    x = torch.arange(output_size, device=device, dtype=dtype) - center
    y = torch.arange(output_size, device=device, dtype=dtype) - center
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    max_profile_radius = len(out) - 1
    grid_x = (R / max_profile_radius) * 2.0 - 1.0
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    profile_reshaped = out.view(1, 1, 1, -1)
    reconstructed_2d = F.grid_sample(
        profile_reshaped, 
        grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    )
    
    return reconstructed_2d.squeeze()


def compute_psd_from_tensor(img_tensor):
    with torch.no_grad():
        x = img_tensor.squeeze(1)
        slice_ft = torch.fft.fftshift(torch.fft.fft2(x))
        psd = torch.abs(slice_ft) ** 2
        psd = torch.log(psd + 1)
        psd_min = psd.min()
        psd_max = psd.max()
        psd = (psd - psd_min) / (psd_max - psd_min + 1e-10)
        return psd.unsqueeze(1).float()


def extract_kernel_name(filename):
    if '_filter_' in filename:
        return filename.split('_filter_')[1].split('.')[0]
    return 'unknown'


def reconstruct_volume(sample, model, device, output_dir):
    data_smooth = sample['smooth_volume']
    data_sharp  = sample['sharp_volume']
    volume_id   = sample['volume_id']
    smooth_kernel = extract_kernel_name(sample['smooth_file'])
    sharp_kernel  = extract_kernel_name(sample['sharp_file'])
    num_slices = data_smooth.shape[2]

    vol_generated_sharp  = np.zeros_like(data_smooth, dtype=np.float32)
    vol_generated_smooth = np.zeros_like(data_sharp,  dtype=np.float32)

    for k in range(num_slices):
        s_slice = data_smooth[:, :, k].copy()
        h_slice = data_sharp[:,  :, k].copy()

        s_slice = np.clip(s_slice, -1000, 3000)
        h_slice = np.clip(h_slice, -1000, 3000)

        s_slice_norm = (s_slice + 1000) / 4000 
        h_slice_norm = (h_slice + 1000) / 4000 

        I_smooth_tensor = torch.from_numpy(s_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        I_sharp_tensor  = torch.from_numpy(h_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        cur_smooth_psd = compute_psd_from_tensor(I_smooth_tensor)
        cur_sharp_psd  = compute_psd_from_tensor(I_sharp_tensor)

        with torch.no_grad():
            out_smooth = model(cur_smooth_psd)
            out_sharp = model(cur_sharp_psd)

            H_smooth = model_to_complex_kernel(out_smooth)
            H_sharp  = model_to_complex_kernel(out_sharp)

            filt_s2sh = complex_kernel_ratio(H_sharp,  H_smooth)
            filt_sh2s = complex_kernel_ratio(H_smooth, H_sharp)

            I_gen_sharp  = apply_complex_filter(I_smooth_tensor, filt_s2sh, device)
            I_gen_smooth = apply_complex_filter(I_sharp_tensor,  filt_sh2s, device)

        res_sharp  = I_gen_sharp.detach().cpu().numpy().squeeze()
        res_smooth = I_gen_smooth.detach().cpu().numpy().squeeze()

        res_sharp  = res_sharp.clip(0, 1.0)
        res_smooth = res_smooth.clip(0, 1.0)

        vol_generated_sharp[:,  :, k] = (res_sharp * 4000) - 1000
        vol_generated_smooth[:, :, k] = (res_smooth * 4000) - 1000

        vol_generated_sharp[:,  :, k] = vol_generated_sharp[:,  :, k].clip(-1000, 3000)
        vol_generated_smooth[:, :, k] = vol_generated_smooth[:, :, k].clip(-1000, 3000)

    nii_generated_sharp  = nib.Nifti1Image(vol_generated_sharp,  sample['sharp_affine'],  sample['sharp_header'])
    nii_generated_smooth = nib.Nifti1Image(vol_generated_smooth, sample['smooth_affine'], sample['smooth_header'])

    sharp_output_path  = os.path.join(output_dir, f'{volume_id}_{sharp_kernel}_to_{smooth_kernel}.nii.gz')
    smooth_output_path = os.path.join(output_dir, f'{volume_id}_{smooth_kernel}_to_{sharp_kernel}.nii.gz')

    nib.save(nii_generated_sharp,  sharp_output_path)
    nib.save(nii_generated_smooth, smooth_output_path)

    print(f'Saved: {os.path.basename(sharp_output_path)} and {os.path.basename(smooth_output_path)}')


for idx in range(len(dataset)):
    print(f'\nProcessing volume {idx+1}/{len(dataset)}')
    sample = dataset[idx]
    reconstruct_volume(sample, model, device, output_dir)

print(f'\nReconstruction complete! All files saved to: {output_dir}')
