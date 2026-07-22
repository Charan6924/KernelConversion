import torch
import matplotlib.pyplot as plt
import numpy as np
from models.SplineEstimator import KernelEstimator
from data.Dataset import MTFPSDDataset
from utils.utils import get_torch_spline
import os

mtf_folder = r'/home/cxv166/PhantomTesting/MTF_Results_Output'
psd_folder = r'/home/cxv166/PhantomTesting/PSD_Results_Output'
plots_dir = r'/home/cxv166/PhantomTesting/Code/plots/'
batch_plots_dir = os.path.join(plots_dir, 'mtf_batch_plots')
os.makedirs(batch_plots_dir, exist_ok=True)

KERNEL_TO_IDX = {'B': 0, 'C': 1, 'CB': 2, 'D': 3, 'E': 4, 'YA': 5, 'YB': 6}
IDX_TO_KERNEL = {v: k for k, v in KERNEL_TO_IDX.items()}

dataset = MTFPSDDataset(mtf_folder=mtf_folder, psd_folder=psd_folder)
train_loader, val_loader, test_loader = dataset.build_dataloaders(
    mtf_folder=mtf_folder, psd_folder=psd_folder
)

model = KernelEstimator()
checkpoint_path = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_kernel256/checkpoints/best_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

freqs = np.linspace(0, 1, 64)
saved = 0

for batch_idx, (input_profile, target_mtf, kernel_idx) in enumerate(test_loader):
    with torch.no_grad():
        predicted_mtfs = model(input_profile)

    batch_size = input_profile.shape[0]

    for i in range(batch_size):
        pred   = predicted_mtfs[i].cpu().numpy().squeeze()
        target = target_mtf[i].cpu().numpy().squeeze()
        kernel_name = IDX_TO_KERNEL[kernel_idx[i].item()]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        axes[0].plot(freqs, target, color='steelblue', linewidth=2)
        axes[0].set_title('Target MTF', fontsize=13)
        axes[0].set_xlabel('Spatial Frequency (normalized)')
        axes[0].set_ylabel('MTF')
        axes[0].set_ylim(0, 5)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(freqs, pred, color='tomato', linewidth=2)
        axes[1].set_title('Predicted MTF', fontsize=13)
        axes[1].set_xlabel('Spatial Frequency (normalized)')
        axes[1].set_ylim(0, 5)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f'KernelEstimator — Batch {batch_idx}, Sample {i} (kernel={kernel_name})',
            fontsize=14
        )
        plt.tight_layout()

        fname = f'batch{batch_idx:03d}_sample{i:03d}_kernel{kernel_name}.png'
        fig.savefig(os.path.join(batch_plots_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved += 1

print(f"Saved {saved} plots to {batch_plots_dir}")
