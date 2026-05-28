from skimage.metrics import peak_signal_noise_ratio
import nibabel as nib
import numpy as np
import glob
import csv
import os

smooth_real_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testA'
smooth_fake_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_2d/testA_fake'
sharp_real_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testB'
sharp_fake_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_2d/testB_fake'

output_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_2d/psnr_results'
os.makedirs(output_dir, exist_ok=True)

smooth_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testA/*.nii', recursive=False))
fake_smooth_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_2d/testA_fake/*.nii', recursive=False))
sharp_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testB/*.nii', recursive=False))
fake_sharp_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_2d/testB_fake/*.nii', recursive=False))

assert smooth_files, "No smooth real files found!"
assert sharp_files, "No sharp real files found!"
assert fake_smooth_files, "No fake smooth files found!"
assert fake_sharp_files, "No fake sharp files found!"

all_smooth_psnrs = []
all_sharp_psnrs = []
summary_csv_path = os.path.join(output_dir, 'summary_psnr.csv')
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

    smooth_csv_path = os.path.join(output_dir, f'{smooth_basename}_psnr.csv')
    sharp_csv_path  = os.path.join(output_dir, f'{sharp_basename}_psnr.csv')

    volume_smooth_psnrs = []
    volume_sharp_psnrs  = []

    with open(smooth_csv_path, 'w', newline='') as smooth_csv:
        smooth_writer = csv.writer(smooth_csv)
        smooth_writer.writerow(['slice_index', 'psnr'])
        for i in range(smooth_volume_real_data.shape[2]):
            psnr = peak_signal_noise_ratio(
                smooth_volume_real_data[:, :, i],
                smooth_volume_fake_data[:, :, i],
                data_range=smooth_volume_real_data.max() - smooth_volume_real_data.min()
            )
            smooth_writer.writerow([i, psnr])
            volume_smooth_psnrs.append(psnr)

    with open(sharp_csv_path, 'w', newline='') as sharp_csv:
        sharp_writer = csv.writer(sharp_csv)
        sharp_writer.writerow(['slice_index', 'psnr'])
        for i in range(sharp_volume_real_data.shape[2]):
            psnr = peak_signal_noise_ratio(
                sharp_volume_real_data[:, :, i],
                sharp_volume_fake_data[:, :, i],
                data_range=sharp_volume_real_data.max() - sharp_volume_real_data.min()
            )
            sharp_writer.writerow([i, psnr])
            volume_sharp_psnrs.append(psnr)

    print(f'Saved per-slice CSVs:\n  {smooth_csv_path}\n  {sharp_csv_path}')

    vol_smooth_avg = np.mean(volume_smooth_psnrs)
    vol_sharp_avg  = np.mean(volume_sharp_psnrs)

    summary_rows.append([smooth_basename, 'smooth', vol_smooth_avg, len(volume_smooth_psnrs)])
    summary_rows.append([sharp_basename,  'sharp',  vol_sharp_avg,  len(volume_sharp_psnrs)])

    all_smooth_psnrs.extend(volume_smooth_psnrs)
    all_sharp_psnrs.extend(volume_sharp_psnrs)

    print(f'[{smooth_basename}] smooth avg PSNR: {vol_smooth_avg:.4f}')
    print(f'[{sharp_basename}]  sharp avg PSNR:  {vol_sharp_avg:.4f}')

overall_smooth_avg = np.mean(all_smooth_psnrs)
overall_sharp_avg  = np.mean(all_sharp_psnrs)

with open(summary_csv_path, 'w', newline='') as summary_csv:
    summary_writer = csv.writer(summary_csv)
    summary_writer.writerow(['volume', 'type', 'avg_psnr', 'num_slices'])
    summary_writer.writerows(summary_rows)
    summary_writer.writerow(['OVERALL', 'smooth', overall_smooth_avg, len(all_smooth_psnrs)])
    summary_writer.writerow(['OVERALL', 'sharp',  overall_sharp_avg,  len(all_sharp_psnrs)])

print(f'smooth PSNR: {overall_smooth_avg}')
print(f'sharp PSNR: {overall_sharp_avg}')
print(f'saved to: {summary_csv_path}')
