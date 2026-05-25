import numpy as np
import nibabel as nib
import torch
import os
from torch.utils.data import DataLoader
from SplineEstimator import KernelEstimator
from utils import generate_images, spline_to_kernel, get_torch_spline
from TestDataset import TestDataset
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KernelEstimator()
checkpoint = torch.load("/home/cxv166/PhantomTesting/Code/training_output_kernel2d/checkpoints/best_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() 
print('Loaded model successfully')

data_root = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
dataset = TestDataset(root_dir=data_root, preload=True)
print('Loaded test dataset')

output_dir = "/home/cxv166/PhantomTesting/reconstructions"
os.makedirs(output_dir, exist_ok=True)
plot_dir   = os.path.join(output_dir, "model_output_plots")
os.makedirs(plot_dir, exist_ok=True)

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

def plot_model_outputs(out_smooth, out_sharp, volume_id, smooth_kernel,
                       sharp_kernel, slice_idx, plot_dir):
    """
    Save a figure with the raw 256-point model outputs for one slice.

    Parameters
    ----------
    out_smooth, out_sharp : torch.Tensor  shape (1, 256)
    """
    y_smooth = out_smooth.squeeze().cpu().numpy()   # (256,)
    y_sharp  = out_sharp.squeeze().cpu().numpy()    # (256,)
    x        = np.arange(len(y_smooth))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    axes[0].plot(x, y_smooth, color='steelblue', linewidth=1.2)
    axes[0].set_title(f'Model output — smooth input ({smooth_kernel})')
    axes[0].set_xlabel('Output index')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, y_sharp, color='tomato', linewidth=1.2)
    axes[1].set_title(f'Model output — sharp input ({sharp_kernel})')
    axes[1].set_xlabel('Output index')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Volume {volume_id} | Slice {slice_idx}', fontsize=13)
    plt.tight_layout()

    fname = os.path.join(plot_dir,
                         f'{volume_id}_slice{slice_idx:03d}_model_outputs.png')
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return fname


def reconstruct_volume(sample, model, device, output_dir, plot_dir):
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
            smooth_knots, smooth_control = model(cur_smooth_psd)
            sharp_knots, sharp_control = model(cur_sharp_psd)

            out_smooth = get_torch_spline(knots = smooth_knots, control_points = smooth_control, num_points = 256)
            out_sharp = get_torch_spline(knots = sharp_knots, control_points = sharp_control, num_points = 256)

            filter_smooth, filter_sharp = spline_to_kernel(out_smooth, out_sharp)

            filter_smooth2sharp = filter_sharp/(filter_smooth + 1e-10)
            filter_sharp2smooth = filter_smooth/(filter_sharp + 1e-10)

            I_gen_sharp, I_gen_smooth = generate_images(
                I_smooth=I_smooth_tensor,
                I_sharp=I_sharp_tensor,
                filter_smooth2sharp=filter_smooth2sharp,
                filter_sharp2smooth=filter_sharp2smooth,
                device=device
            )

            if k % 10 == 0:
                plot_model_outputs(
                    out_smooth, out_sharp,
                    volume_id, smooth_kernel, sharp_kernel,
                    slice_idx=k,
                    plot_dir=plot_dir
                )

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

    residual_sharp  = data_sharp  - vol_generated_sharp
    residual_smooth = data_smooth - vol_generated_smooth

    nib.save(
        nib.Nifti1Image(residual_sharp,  sample['sharp_affine'],  sample['sharp_header']),
        os.path.join(output_dir, f'{volume_id}_{smooth_kernel}_to_{sharp_kernel}_residual.nii.gz')
    )
    nib.save(
        nib.Nifti1Image(residual_smooth, sample['smooth_affine'], sample['smooth_header']),
        os.path.join(output_dir, f'{volume_id}_{sharp_kernel}_to_{smooth_kernel}_residual.nii.gz')
    )

    print(f'Saved: {os.path.basename(sharp_output_path)} and {os.path.basename(smooth_output_path)}')


for idx in range(len(dataset)):
    print(f'\nProcessing volume {idx+1}/{len(dataset)}')
    sample = dataset[idx]
    reconstruct_volume(sample, model, device, output_dir, plot_dir)

print(f'\nReconstruction complete! All files saved to: {output_dir}')
