import nibabel as nib
import numpy as np
import glob
import csv
import os
import torch
import torch_fidelity
from torch.utils.data import Dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

smooth_real_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testA'
smooth_fake_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_256/testA_fake'
sharp_real_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testB'
sharp_fake_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_256/testB_fake'

output_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_256/fid_results'
os.makedirs(output_dir, exist_ok=True)

smooth_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testA/*.nii', recursive=False))
fake_smooth_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_256/testA_fake/*.nii.gz', recursive=False))
sharp_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testB/*.nii', recursive=False))
fake_sharp_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_256/testB_fake/*.nii.gz', recursive=False))

assert smooth_files, "No smooth real files found!"
assert sharp_files, "No sharp real files found!"
assert fake_smooth_files, "No fake smooth files found!"
assert fake_sharp_files, "No fake sharp files found!"


class SliceDataset(Dataset):
    def __init__(self, slices: list[np.ndarray]):
        tensors = []
        for s in slices:
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                s_norm = (s - s_min) / (s_max - s_min)
            else:
                s_norm = np.zeros_like(s)
            s_uint8 = (s_norm * 255).astype(np.uint8)
            t = torch.from_numpy(s_uint8).unsqueeze(0).repeat(3, 1, 1)
            tensors.append(t)
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]


def compute_fid(real_slices, fake_slices):
    real_ds = SliceDataset(real_slices)
    fake_ds = SliceDataset(fake_slices)
    metrics = torch_fidelity.calculate_metrics(
        input1=real_ds,
        input2=fake_ds,
        fid=True,
        verbose=False,
    )
    return metrics['frechet_inception_distance']


all_smooth_real_slices = []
all_smooth_fake_slices = []
all_sharp_real_slices = []
all_sharp_fake_slices = []

summary_csv_path = os.path.join(output_dir, 'summary_fid.csv')
summary_rows = []

for smooth_file, sharp_file, fake_smooth_file, fake_sharp_file in zip(
    smooth_files, sharp_files, fake_smooth_files, fake_sharp_files
):
    smooth_volume_real_data = nib.load(smooth_file).get_fdata()
    smooth_volume_fake_data = nib.load(fake_smooth_file).get_fdata()
    sharp_volume_real_data  = nib.load(sharp_file).get_fdata()
    sharp_volume_fake_data  = nib.load(fake_sharp_file).get_fdata()

    smooth_basename = os.path.basename(smooth_file).replace('.nii', '').replace('.gz', '')
    sharp_basename  = os.path.basename(sharp_file).replace('.nii', '').replace('.gz', '')

    vol_smooth_real_slices = []
    vol_smooth_fake_slices = []
    vol_sharp_real_slices  = []
    vol_sharp_fake_slices  = []

    for i in range(smooth_volume_real_data.shape[2]):
        vol_smooth_real_slices.append(smooth_volume_real_data[:, :, i])
        vol_smooth_fake_slices.append(smooth_volume_fake_data[:, :, i])

    for i in range(sharp_volume_real_data.shape[2]):
        vol_sharp_real_slices.append(sharp_volume_real_data[:, :, i])
        vol_sharp_fake_slices.append(sharp_volume_fake_data[:, :, i])

    vol_smooth_fid = compute_fid(vol_smooth_real_slices, vol_smooth_fake_slices)
    vol_sharp_fid  = compute_fid(vol_sharp_real_slices,  vol_sharp_fake_slices)

    summary_rows.append([smooth_basename, 'smooth', len(vol_smooth_real_slices), vol_smooth_fid])
    summary_rows.append([sharp_basename,  'sharp',  len(vol_sharp_real_slices),  vol_sharp_fid])

    all_smooth_real_slices.extend(vol_smooth_real_slices)
    all_smooth_fake_slices.extend(vol_smooth_fake_slices)
    all_sharp_real_slices.extend(vol_sharp_real_slices)
    all_sharp_fake_slices.extend(vol_sharp_fake_slices)

    print(f'[{smooth_basename}] smooth FID: {vol_smooth_fid:.4f}')
    print(f'[{sharp_basename}]  sharp FID:  {vol_sharp_fid:.4f}')

overall_smooth_fid = compute_fid(all_smooth_real_slices, all_smooth_fake_slices)
overall_sharp_fid  = compute_fid(all_sharp_real_slices,  all_sharp_fake_slices)

with open(summary_csv_path, 'w', newline='') as summary_csv:
    summary_writer = csv.writer(summary_csv)
    summary_writer.writerow(['volume', 'type', 'num_slices', 'fid'])
    summary_writer.writerows(summary_rows)
    summary_writer.writerow(['OVERALL', 'smooth', len(all_smooth_real_slices), overall_smooth_fid])
    summary_writer.writerow(['OVERALL', 'sharp',  len(all_sharp_real_slices),  overall_sharp_fid])

print(f'smooth FID: {overall_smooth_fid:.4f}')
print(f'sharp  FID: {overall_sharp_fid:.4f}')
print(f'saved to: {summary_csv_path}')
