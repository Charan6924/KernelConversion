from skimage.metrics import structural_similarity
import nibabel as nib
import numpy as np
import glob
import csv
import os

smooth_real_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testA'
smooth_fake_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_pix2pix/testA_fake'
sharp_real_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testB'
sharp_fake_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_pix2pix/testB_fake'

output_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/eval_pix2pix/ssim_results'
os.makedirs(output_dir, exist_ok=True)

smooth_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testA/*.nii', recursive=False))
fake_smooth_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_pix2pix/testA_fake/*.nii.gz', recursive=False))
sharp_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/testB/*.nii', recursive=False))
fake_sharp_files = sorted(glob.glob('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_pix2pix/testB_fake/*.nii.gz', recursive=False))

assert smooth_files, "No smooth real files found!"
assert sharp_files, "No sharp real files found!"
assert fake_smooth_files, "No fake smooth files found!"
assert fake_sharp_files, "No fake sharp files found!"

all_smooth_ssims = []
all_sharp_ssims = []
summary_csv_path = os.path.join(output_dir, 'summary_ssim.csv')
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

    smooth_csv_path = os.path.join(output_dir, f'{smooth_basename}_ssim.csv')
    sharp_csv_path  = os.path.join(output_dir, f'{sharp_basename}_ssim.csv')

    volume_smooth_ssims = []
    volume_sharp_ssims  = []

    with open(smooth_csv_path, 'w', newline='') as smooth_csv:
        smooth_writer = csv.writer(smooth_csv)
        smooth_writer.writerow(['slice_index', 'ssim'])
        for i in range(smooth_volume_real_data.shape[2]):
            ssim = structural_similarity(
                smooth_volume_real_data[:, :, i],
                smooth_volume_fake_data[:, :, i],
                data_range=smooth_volume_real_data.max() - smooth_volume_real_data.min()
            )
            smooth_writer.writerow([i, ssim])
            volume_smooth_ssims.append(ssim)

    with open(sharp_csv_path, 'w', newline='') as sharp_csv:
        sharp_writer = csv.writer(sharp_csv)
        sharp_writer.writerow(['slice_index', 'ssim'])
        for i in range(sharp_volume_real_data.shape[2]):
            ssim = structural_similarity(
                sharp_volume_real_data[:, :, i],
                sharp_volume_fake_data[:, :, i],
                data_range=sharp_volume_real_data.max() - sharp_volume_real_data.min()
            )
            sharp_writer.writerow([i, ssim])
            volume_sharp_ssims.append(ssim)

    print(f'Saved per-slice CSVs:\n  {smooth_csv_path}\n  {sharp_csv_path}')

    vol_smooth_avg = np.mean(volume_smooth_ssims)
    vol_sharp_avg  = np.mean(volume_sharp_ssims)

    summary_rows.append([smooth_basename, 'smooth', vol_smooth_avg, len(volume_smooth_ssims)])
    summary_rows.append([sharp_basename,  'sharp',  vol_sharp_avg,  len(volume_sharp_ssims)])

    all_smooth_ssims.extend(volume_smooth_ssims)
    all_sharp_ssims.extend(volume_sharp_ssims)

    print(f'[{smooth_basename}] smooth avg SSIM: {vol_smooth_avg:.4f}')
    print(f'[{sharp_basename}]  sharp avg SSIM:  {vol_sharp_avg:.4f}')

overall_smooth_avg = np.mean(all_smooth_ssims)
overall_sharp_avg  = np.mean(all_sharp_ssims)

with open(summary_csv_path, 'w', newline='') as summary_csv:
    summary_writer = csv.writer(summary_csv)
    summary_writer.writerow(['volume', 'type', 'avg_ssim', 'num_slices'])
    summary_writer.writerows(summary_rows)
    summary_writer.writerow(['OVERALL', 'smooth', overall_smooth_avg, len(all_smooth_ssims)])
    summary_writer.writerow(['OVERALL', 'sharp',  overall_sharp_avg,  len(all_sharp_ssims)])

print(f'\nOverall smooth SSIM: {overall_smooth_avg:.4f}')
print(f'Overall sharp  SSIM: {overall_sharp_avg:.4f}')
print(f'Summary saved to: {summary_csv_path}')
