import numpy as np
import nibabel as nib
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import spline_to_kernel, generate_images
from TestDataset import TestDataset
from KernelEstimator import KernelEstimator
from utils import compute_fft

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KernelEstimator()
checkpoint = torch.load("/home/cxv166/PhantomTesting/Code/training_output_0.5/checkpoints/epoch_7_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print('Loaded model successfully')

data_root = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
dataset = TestDataset(root_dir=data_root, preload=True)
print('Loaded test dataset')

output_dir = "/home/cxv166/PhantomTesting/reconstructions"
plot_dir   = os.path.join(output_dir, "model_output_plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir,   exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── main reconstruction ───────────────────────────────────────────────────────

def reconstruct_volume(sample, model, device, output_dir, plot_dir,
                       plot_every_n_slices=10):
    """
    Reconstruct a volume and save plots of the model outputs every
    `plot_every_n_slices` slices.
    """
    data_smooth   = sample['smooth_volume']
    data_sharp    = sample['sharp_volume']
    volume_id     = sample['volume_id']
    smooth_kernel = extract_kernel_name(sample['smooth_file'])
    sharp_kernel  = extract_kernel_name(sample['sharp_file'])
    num_slices    = data_smooth.shape[2]

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

        I_smooth_fft = compute_fft(I_smooth_tensor)
        I_sharp_fft  = compute_fft(I_sharp_tensor)

        with torch.no_grad():
            out_smooth = model(cur_smooth_psd)  # (1, 256)
            out_sharp  = model(cur_sharp_psd)   # (1, 256)

        # ── plot model outputs for selected slices ──────────────────────────
        if k % plot_every_n_slices == 0:
            plot_model_outputs(
                out_smooth, out_sharp,
                volume_id, smooth_kernel, sharp_kernel,
                slice_idx=k,
                plot_dir=plot_dir
            )

        # ── TODO: build filter_s2sh / filter_sh2s from out_smooth / out_sharp
        #    before using them below (currently undefined).
        # filter_s2sh = ...
        # filter_sh2s = ...

        
    print(f'Plots saved to: {plot_dir}')


# ── run ───────────────────────────────────────────────────────────────────────

for idx in range(len(dataset)):
    print(f'\nProcessing volume {idx+1}/{len(dataset)}')
    sample = dataset[idx]
    reconstruct_volume(
        sample, model, device,
        output_dir=output_dir,
        plot_dir=plot_dir,
        plot_every_n_slices=10   # ← change to 1 to plot every slice
    )

print(f'\nReconstruction complete! All files saved to: {output_dir}')
print(f'Model output plots saved to: {plot_dir}')
