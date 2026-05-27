import torch
import torch.nn as nn
import torch.nn.functional as F
from Dataset import MTFPSDDataset
from SplineEstimator import KernelEstimator
from utils import get_torch_spline
import os
import matplotlib.pyplot as plt
import numpy as np

model = KernelEstimator()
model.to('cuda')
checkpoint = torch.load('/home/cxv166/PhantomTesting/Code/training_output_kernel2d/checkpoints/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print('loaded model')
mtf_dataset = MTFPSDDataset(mtf_folder = '/home/cxv166/PhantomTesting/MTF_Results_Output',psd_folder = '/home/cxv166/PhantomTesting/PSD_Results_Output')
dict = {'B': 0, 'C': 1, 'CB': 2, 'D': 3, 'E': 4, 'YA': 5, 'YB': 6}
rev = {v: k for k, v in dict.items()} 
_,_,test_loader = mtf_dataset.build_dataloaders(mtf_folder = '/home/cxv166/PhantomTesting/MTF_Results_Output',psd_folder = '/home/cxv166/PhantomTesting/PSD_Results_Output')
print('created dataloader')
os.makedirs('plots', exist_ok=True)


input_profile, target_mtf, kernel_idx = next(iter(test_loader))
input_profile = input_profile.to('cuda')
target_mtf = target_mtf.to('cuda')
kernel_idx = kernel_idx.to('cuda')
knots, control_points = model(input_profile)
mtf = get_torch_spline(knots,control_points,num_points=64)

mtf_cpu        = mtf.detach().cpu().numpy()          # (24, 64)
target_mtf_cpu = target_mtf.detach().cpu().numpy()   # (24, 64)
kernel_idx_cpu = kernel_idx.detach().cpu().numpy()   # (24,)

x = np.linspace(0, 1, 64)

for i in range(len(mtf_cpu)):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    kernel_name = rev.get(int(kernel_idx_cpu[i]), f"idx_{int(kernel_idx_cpu[i])}")
    fig.suptitle(f"Sample {i}  |  Kernel: {kernel_name}", fontsize=13, fontweight='bold')

    # --- Predicted MTF ---
    axes[0].plot(x, mtf_cpu[i].squeeze()[::-1], color='steelblue', linewidth=2)
    axes[0].set_title("Predicted MTF")
    axes[0].set_xlabel("Normalised Spatial Frequency")
    axes[0].set_ylabel("MTF")
    axes[0].set_ylim(0, 3)
    axes[0].grid(True, alpha=0.3)

    # --- Target MTF ---
    axes[1].plot(x, target_mtf_cpu[i], color='tomato', linewidth=2)
    axes[1].set_title("Target MTF")
    axes[1].set_xlabel("Normalised Spatial Frequency")
    axes[1].set_ylabel("MTF")
    axes[1].set_ylim(0, 3)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"plots/sample_{i:03d}_kernel_{kernel_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

print(f"\nAll {len(mtf_cpu)} plots saved to ./plots/")
