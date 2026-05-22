from KernelEstimator import KernelEstimator
from filterModel import FilterEstimator
from utils import spline_to_kernel, get_torch_spline, generate_images, compute_fft, compute_psd
from PSDDataset import PSDDataset
from torch.utils.data import DataLoader
from scipy.ndimage import uniform_filter1d
import torch
import matplotlib.pyplot as plt

device = 'cuda'

# --- FilterEstimator ---
model = FilterEstimator()
checkpoint = torch.load('/home/cxv166/KernelConversionResearch/training_filter_model/checkpoints/epoch_17.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

dataset = PSDDataset(root_dir=r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root")
loader  = DataLoader(dataset=dataset, batch_size=32)
I_smooth, I_sharp, _, _ = next(iter(loader))

psd_smooth = compute_psd(I_smooth, device='cuda').to(device, non_blocking=True)
psd_sharp  = compute_psd(I_sharp,  device='cuda').to(device, non_blocking=True)

with torch.no_grad():
    filters_s2sh, filters_sh2s = model(psd_smooth, psd_sharp)

# --- KernelEstimator ---
kernel_model = KernelEstimator()
checkpoint = torch.load('/home/cxv166/PhantomTesting/Code/training_output_kernel256/checkpoints/best_checkpoint.pth', map_location=device)
kernel_model.load_state_dict(checkpoint['model_state_dict'])
kernel_model.to(device)
kernel_model.eval()

with torch.no_grad():
    out_sharp  = kernel_model(psd_sharp)
    out_smooth = kernel_model(psd_smooth)

out_sharp_np  = out_sharp[0].detach().cpu().numpy()
out_smooth_np = out_smooth[0].detach().cpu().numpy()
f_s2sh_np     = filters_s2sh[0].detach().cpu().numpy()
f_sh2s_np     = filters_sh2s[0].detach().cpu().numpy()

# Smooth
out_sharp_smooth  = uniform_filter1d(out_sharp_np,  size=11)
out_smooth_smooth = uniform_filter1d(out_smooth_np, size=11)

# --- MTFs side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

ax1.plot(out_sharp_smooth, color='steelblue')
ax1.set_title('MTF Sharp')
ax1.set_xlabel('Frequency bin')
ax1.set_ylabel('MTF')
ax1.grid(True, alpha=0.3)

ax2.plot(out_smooth_smooth, color='tomato')
ax2.set_title('MTF Smooth')
ax2.set_xlabel('Frequency bin')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Mtf_comparison.png', dpi=150)
plt.clf()

# --- Filter plots ---
plt.imshow(f_s2sh_np, cmap='hot')
plt.title('Filter smooth to sharp')
plt.colorbar()
plt.savefig('s2sh.png')
plt.clf()

plt.imshow(f_sh2s_np, cmap='hot')
plt.title('Filter sharp to smooth')
plt.colorbar()
plt.savefig('sh2s.png')
plt.clf()
