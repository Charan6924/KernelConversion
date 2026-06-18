'''measuring ground truth mtf ratios with model predicted ratios'''

from data.Dataset import MTFPSDDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.SplineEstimator import KernelEstimator
from utils.utils import compute_psd,compute_fft, generate_images, spline_to_kernel, get_torch_spline
import os
import pydicom
from scipy.interpolate import interp1d
from scipy.io import loadmat

model = KernelEstimator()
model.to('cuda')
checkpoint = torch.load('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_spline/checkpoints/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print('loaded model')

mtf_e = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_B_MTF_Results_mat.mat')
mtf_d = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_C_MTF_Results_mat.mat')

results = mtf_e['results']
r = results[0, 0]
mtf_axis_e  = r['mtfAxis'][0]    # shape (64,)
mtf_val_e   = r['mtfVal'][0]     # shape (64,)

results = mtf_d['results']
r = results[0, 0]
mtf_axis_d  = r['mtfAxis'][0]    # shape (64,)
mtf_val_d   = r['mtfVal'][0]  

f_e = interp1d(mtf_axis_e, mtf_val_e, kind='cubic')
f_d = interp1d(mtf_axis_d, mtf_val_d, kind='cubic')

freq_min = max(mtf_axis_e[0], mtf_axis_d[0])
freq_max = min(mtf_axis_e[-1], mtf_axis_d[-1])

mtf_axis_256 = np.linspace(freq_min, freq_max, 256)
mtf_val_e_256 = torch.tensor(f_e(mtf_axis_256), dtype=torch.float32).unsqueeze(0)
mtf_val_d_256 = torch.tensor(f_d(mtf_axis_256), dtype=torch.float32).unsqueeze(0)

kernel_e, kernel_d = spline_to_kernel(mtf_val_e_256, mtf_val_d_256)

filterEtoD = kernel_d/(kernel_e + 1e-10)
filterDtoE = kernel_e/(kernel_d + 1e-10)

psd_d = torch.tensor(np.load('/home/cxv166/PhantomTesting/PSD_Results_Output/I20_Kernel_B_PSD.npy'),dtype=torch.float32).to('cuda')
psd_e = torch.tensor(np.load('/home/cxv166/PhantomTesting/PSD_Results_Output/I20_Kernel_C_PSD.npy'), dtype = torch.float32).to('cuda')
psd_d = psd_d.unsqueeze(0).unsqueeze(0)
psd_e = psd_e.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    e_knots, e_cp = model(psd_e)
    d_knots, d_cp = model(psd_d)

    mtf_e_predicted = get_torch_spline(e_knots,e_cp,num_points = 256)
    mtf_d_predicted = get_torch_spline(d_knots, d_cp,num_points = 256)

filterEtoD = mtf_val_d_256/(mtf_val_e_256+1e-10)
filterDtoE = mtf_val_e_256/(mtf_val_d_256 + 1e-10)

filterEtoD_predicted = mtf_d_predicted/(mtf_e_predicted + 1e-10)
filterDtoE_predicted = mtf_e_predicted/(mtf_d_predicted + 1e-10)

freq = mtf_axis_256  # already defined above, shape (256,)

#plot all 4 filters
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

plots = [
    (filterEtoD.numpy().squeeze(),       'B→C (ground truth)',    'B→C GT',    '#3266ad'),
    (filterDtoE.numpy().squeeze(),       'C→B (ground truth)',    'C→B GT',    '#c0392b'),
    (filterEtoD_predicted.cpu().numpy().squeeze(), 'B→B (predicted)', 'B→C Pred', '#1a9e75'),
    (filterDtoE_predicted.cpu().numpy().squeeze(), 'C→B (predicted)', 'C→B Pred', '#8e44ad'),
]

for ax, (vals, title, label, color) in zip(axes, plots):
    ax.plot(freq, vals, color=color, linewidth=1.8, label=label)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Spatial frequency (cyc/mm)', fontsize=10)
    ax.set_ylabel('Filter magnitude', fontsize=10)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('MTF Ratio Filters: Ground Truth vs Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mtf_filters.png', dpi=150, bbox_inches='tight')
plt.show()
