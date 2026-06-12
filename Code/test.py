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

mtf_e = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_CB_MTF_Results_mat.mat')
mtf_d = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_YA_MTF_Results_mat.mat')

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

psd_d = torch.tensor(np.load('/home/cxv166/PhantomTesting/PSD_Results_Output/I20_Kernel_CB_PSD.npy'),dtype=torch.float32).to('cuda')
psd_e = torch.tensor(np.load('/home/cxv166/PhantomTesting/PSD_Results_Output/I20_Kernel_YA_PSD.npy'), dtype = torch.float32).to('cuda')
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

def to_np(t):
    return t.squeeze().detach().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(freq, to_np(mtf_val_e_256),      label="GT",        lw=1.5)
axes[0].plot(freq, to_np(mtf_e_predicted),    label="predicted", lw=1.5, ls="--")
axes[0].set_title("MTF — kernel E"); axes[0].legend()
axes[0].set_xlabel("spatial frequency (cyc/mm)"); axes[0].set_ylabel("MTF")
axes[0].grid(True, lw=0.4, alpha=0.5)

axes[1].plot(freq, to_np(mtf_val_d_256),      label="GT",        lw=1.5)
axes[1].plot(freq, to_np(mtf_d_predicted),    label="predicted", lw=1.5, ls="--")
axes[1].set_title("MTF — kernel D"); axes[1].legend()
axes[1].set_xlabel("spatial frequency (cyc/mm)"); axes[1].set_ylabel("MTF")
axes[1].grid(True, lw=0.4, alpha=0.5)

plt.tight_layout()
plt.savefig("mtf_direct_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
