import numpy as np
import nibabel as nib
import torch
import os
from torch.utils.data import DataLoader
from SplineEstimator import KernelEstimator
from utils import generate_images, spline_to_kernel 
from TestDataset import TestDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KernelEstimator()
checkpoint = torch.load(r"D:\Charan work file\PhantomTesting\Code\training_output_0.5\checkpoints\best_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() 
print('Loaded model successfully')

data_root = r"D:\Charan work file\KernelEstimator\Data_Root"
dataset = TestDataset(root_dir=data_root, preload=True)
print('Loaded test dataset')

output_dir = r"D:\Charan work file\PhantomTesting\reconstructions"
os.makedirs(output_dir, exist_ok=True)


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
            sharp_knots, sharp_control = model(cur_smooth_psd)
            smooth_knots,  smooth_control  = model(cur_sharp_psd)

            filter_smooth2sharp, filter_sharp2smooth = spline_to_kernel(smooth_knots=smooth_knots,smooth_control_points=smooth_control,sharp_knots=sharp_knots,sharp_control_points=sharp_control,grid_size=512)
            I_gen_sharp,  I_gen_smooth = generate_images(I_smooth=I_smooth_tensor,I_sharp=I_sharp_tensor,filter_smooth2sharp=filter_smooth2sharp,filter_sharp2smooth=filter_sharp2smooth,device=device)

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

    sharp_output_path  = os.path.join(output_dir, f'{volume_id}_{smooth_kernel}_to_{sharp_kernel}.nii.gz')
    smooth_output_path = os.path.join(output_dir, f'{volume_id}_{sharp_kernel}_to_{smooth_kernel}.nii.gz')

    nib.save(nii_generated_sharp,  sharp_output_path)
    nib.save(nii_generated_smooth, smooth_output_path)

    print(f'Saved: {os.path.basename(sharp_output_path)} and {os.path.basename(smooth_output_path)}')


for idx in range(len(dataset)):
    print(f'\nProcessing volume {idx+1}/{len(dataset)}')
    sample = dataset[idx]
    reconstruct_volume(sample, model, device, output_dir)

print(f'\nReconstruction complete! All files saved to: {output_dir}')