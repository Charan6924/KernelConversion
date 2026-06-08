import numpy as np
import nibabel as nib
import torch
import os
import matplotlib
from torch.utils.data import DataLoader
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.utils import compute_psd, spline_to_kernel, generate_images
from data.Dataset import MTFPSDDataset
from data.PSDDataset import PSDDataset
from models.KernelEstimator import KernelEstimator
from utils.utils import compute_fft, spline_to_kernel
from scipy.interpolate import CubicSpline

'''
Reconstructing patient volumes with ground truth phantom measurements
'''

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = KernelEstimator()
# checkpoint = torch.load("/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_kernel256/checkpoints/best_checkpoint.pth", map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)
# model.eval()
# print('Loaded model successfully')

KERNEL_TO_IDX = {'B': 0, 'C': 1, 'CB': 2, 'D': 3, 'E': 4, 'YA': 5, 'YB': 6}
IDX_TO_KERNEL = {v: k for k, v in KERNEL_TO_IDX.items()}

mtf_folder = r'/home/cxv166/PhantomTesting/MTF_Results_Output'
psd_folder = r'/home/cxv166/PhantomTesting/PSD_Results_Output'

dataset = MTFPSDDataset(mtf_folder=mtf_folder, psd_folder=psd_folder)
train_loader, val_loader, test_loader = dataset.build_dataloaders(
    mtf_folder=mtf_folder, psd_folder=psd_folder
)
device = 'cuda'
smooth = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testA/MC0O45BIX5_filter_B.nii'
sharp  = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testB/MC0O45BIX5_filter_E.nii'

smooth_nib = nib.load(smooth)
sharp_nib  = nib.load(sharp)

smooth_affine = smooth_nib.affine
smooth_header = smooth_nib.header
sharp_affine  = sharp_nib.affine
sharp_header  = sharp_nib.header

smooth_vol_np = smooth_nib.get_fdata()
sharp_vol_np  = sharp_nib.get_fdata()

smooth_vol_tensor = torch.from_numpy(smooth_vol_np)
sharp_vol_tensor  = torch.from_numpy(sharp_vol_np)

for batch_idx, (input_profile, target_mtf, kernel_idx) in enumerate(test_loader):

    b_idx  = KERNEL_TO_IDX['B']
    e_idx  = KERNEL_TO_IDX['E']

    b_mask = (kernel_idx == b_idx)
    e_mask = (kernel_idx == e_idx)

    if b_mask.any() and e_mask.any():
        b_ground_truth = target_mtf[b_mask][0].squeeze().cpu().numpy()  # sharp
        e_ground_truth = target_mtf[e_mask][0].squeeze().cpu().numpy()  # smooth

        n_original = len(b_ground_truth)
        x_orig  = np.linspace(0, 1, n_original)
        x_dense = np.linspace(0, 1, 512)

        b_mtf_512 = CubicSpline(x_orig, b_ground_truth)(x_dense)
        e_mtf_512 = CubicSpline(x_orig, e_ground_truth)(x_dense)

        b_mtf_512_t = torch.from_numpy(b_mtf_512).float().unsqueeze(0) 
        e_mtf_512_t = torch.from_numpy(e_mtf_512).float().unsqueeze(0) 

        b_kernel, e_kernel = spline_to_kernel(b_mtf_512_t, e_mtf_512_t)
        b_kernel = b_kernel.squeeze(0)
        e_kernel = e_kernel.squeeze(0)
        filter_smooth2sharp = b_kernel / (e_kernel + 1e-10)
        filter_sharp2smooth = e_kernel/(b_kernel + 1e-10)
        print(filter_sharp2smooth.shape)

        break


vol_generated_sharp  = np.zeros_like(sharp_vol_np, dtype=np.float32)
vol_generated_smooth = np.zeros_like(smooth_vol_np,  dtype=np.float32)

for i in range(vol_generated_smooth.shape[2]):
    I_smooth = smooth_vol_tensor[:,:,i]
    print(I_smooth.shape)
    I_sharp = sharp_vol_tensor[:,:,i]

    filter_smooth2sharp = filter_smooth2sharp.to('cuda')
    filter_sharp2smooth = filter_sharp2smooth.to('cuda')
    
    I_gen_sharp, I_gen_smooth = generate_images(
                I_smooth=I_smooth,
                I_sharp=I_sharp,
                filter_smooth2sharp=filter_smooth2sharp,
                filter_sharp2smooth=filter_sharp2smooth,
                device=device
            )
    res_sharp  = I_gen_sharp.detach().cpu().numpy().squeeze()
    res_smooth = I_gen_smooth.detach().cpu().numpy().squeeze()

    res_sharp  = res_sharp.clip(0, 1.0)
    res_smooth = res_smooth.clip(0, 1.0)

    vol_generated_sharp[:,  :, i] = ((res_sharp  * 4000) - 1000).clip(-1000, 3000)
    vol_generated_smooth[:, :, i] = ((res_smooth * 4000) - 1000).clip(-1000, 3000)

    smooth_fname = f'EtoB.nii.gz'
    sharp_fname  = f'BtoE.nii.gz'

nib.save(
        nib.Nifti1Image(vol_generated_smooth, smooth_affine, smooth_header),
        smooth_fname
    )
nib.save(
        nib.Nifti1Image(vol_generated_sharp, sharp_affine, sharp_header),
        sharp_fname
    )
print('saved')
