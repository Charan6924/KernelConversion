import os
import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

KERNEL_TO_IDX = {'B': 0, 'C': 1, 'CB': 2, 'D': 3, 'E': 4, 'YA': 5, 'YB': 6}

def extract_kernel_from_filename(filename):
    name = filename.replace('.nii.gz', '').replace('.nii', '')
    if '_filter_' in name:
        kernel_str = name.split('_filter_')[-1]
        return kernel_str, KERNEL_TO_IDX.get(kernel_str, -1)
    raise ValueError(f"Cannot extract kernel from filename: {filename}")


def compute_psd_batch_gpu(image_batch, device):
    if image_batch.device.type != device:
        image_batch = image_batch.to(device)
    freq_map = torch.fft.fftshift(torch.fft.fft2(image_batch), dim=(-2, -1))
    psd = torch.abs(freq_map) ** 2
    psd = torch.log(psd + 1)
    b, c, h, w = psd.shape
    psd_flat = psd.view(b, -1)
    p_min = psd_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    p_max = psd_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    psd = (psd - p_min) / (p_max - p_min + 1e-8)
    return psd


class PSDDataset(Dataset):
    def __init__(self, root_dir, min_slice_percentile=0.1, max_slice_percentile=0.9,
                 preload=True, seed=42):
        self.smooth_dir = os.path.join(root_dir, "trainA")
        self.sharp_dir  = os.path.join(root_dir, "trainB")
        self.min_percentile = min_slice_percentile
        self.max_percentile = max_slice_percentile
        self.preload = preload
        np.random.seed(seed)

        print("Finding volume pairs...")
        volume_pairs = self._find_volume_pairs()
        print(f"Found {len(volume_pairs)} volume pairs")

        if self.preload:
            self.volume_cache = self._preload_all_volumes(volume_pairs)
            print(f"Cached {len(self.volume_cache)} unique volumes")
        else:
            self.volume_cache = {}

        self.slice_data = self._build_slice_index(volume_pairs)
        print(f"Total slices: {len(self.slice_data)}")

        if self.preload and self.volume_cache:
            total_bytes = sum(v.nbytes for v in self.volume_cache.values())
            print(f"Memory usage: {total_bytes / (1024**3):.2f} GB")

        self.volume_to_indices = {}
        for i, info in enumerate(self.slice_data):
            path = info['smooth_path']
            if path not in self.volume_to_indices:
                self.volume_to_indices[path] = []
            self.volume_to_indices[path].append(i)

    def _find_volume_pairs(self):
        smooth_files = sorted([f for f in os.listdir(self.smooth_dir)
                                if f.endswith(('.nii', '.nii.gz'))])
        sharp_files  = sorted([f for f in os.listdir(self.sharp_dir)
                                if f.endswith(('.nii', '.nii.gz'))])

        sharp_dict = {
            (f.split("_filter_")[0] if "_filter_" in f else f.split(".")[0]): f
            for f in sharp_files
        }

        volume_pairs = []
        for sfile in smooth_files:
            base_id = (sfile.split("_filter_")[0] if "_filter_" in sfile
                       else sfile.split(".")[0])
            if base_id in sharp_dict:
                volume_pairs.append((sfile, sharp_dict[base_id]))

        return volume_pairs

    def _preload_all_volumes(self, volume_pairs):
        unique_paths = set()
        for sfile, shfile in volume_pairs:
            unique_paths.add(os.path.join(self.smooth_dir, sfile))
            unique_paths.add(os.path.join(self.sharp_dir,  shfile))

        cache = {}
        for path in tqdm(sorted(unique_paths), desc="Loading volumes"):
            try:
                cache[path] = nib.load(path).get_fdata()  # type: ignore
            except Exception as e:
                print(f"\nFailed to load {os.path.basename(path)}: {e}")
        return cache

    def _build_slice_index(self, volume_pairs):
        slice_data = []

        for sfile, shfile in tqdm(volume_pairs, desc="Indexing slices"):
            s_path  = os.path.join(self.smooth_dir, sfile)
            sh_path = os.path.join(self.sharp_dir,  shfile)

            try:
                smooth_kernel_str, smooth_kernel_idx = extract_kernel_from_filename(sfile)
                sharp_kernel_str,  sharp_kernel_idx  = extract_kernel_from_filename(shfile)
            except ValueError as e:
                print(f"Warning: {e} — skipping pair")
                continue

            if smooth_kernel_idx == -1 or sharp_kernel_idx == -1:
                print(f"Warning: unknown kernel in {sfile} or {shfile} — skipping")
                continue

            try:
                if self.preload:
                    n_slices = self.volume_cache[s_path].shape[2]
                else:
                    n_slices = nib.load(s_path).shape[2]  # type: ignore

                start_idx = int(n_slices * self.min_percentile)
                end_idx   = int(n_slices * self.max_percentile)

                for z_idx in range(start_idx, end_idx):
                    slice_data.append({
                        'smooth_path':       s_path,
                        'sharp_path':        sh_path,
                        'slice_idx':         z_idx,
                        'smooth_kernel_str': smooth_kernel_str,
                        'smooth_kernel_idx': smooth_kernel_idx,
                        'sharp_kernel_str':  sharp_kernel_str,
                        'sharp_kernel_idx':  sharp_kernel_idx,
                    })

            except Exception as e:
                print(f"\nFailed to index {os.path.basename(s_path)}: {e}")
                continue

        return slice_data

    def _get_volume(self, path):
        if path in self.volume_cache:
            return self.volume_cache[path]
        vol = nib.load(path).get_fdata()  # type: ignore
        if not self.preload:
            self.volume_cache[path] = vol
        return vol

    def _get_slice_pair(self, idx):
        info   = self.slice_data[idx]
        vol_s  = self._get_volume(info['smooth_path'])
        vol_h  = self._get_volume(info['sharp_path'])
        img_s  = vol_s[:, :, info['slice_idx']].copy()
        img_h  = vol_h[:, :, info['slice_idx']].copy()

        img_s = np.clip(img_s, -1000, 3000)
        img_h = np.clip(img_h, -1000, 3000)
        img_s = (img_s + 1000) / 4000
        img_h = (img_h + 1000) / 4000

        I_smooth = torch.from_numpy(img_s).unsqueeze(0).float()
        I_sharp  = torch.from_numpy(img_h).unsqueeze(0).float()
        return I_smooth, I_sharp

    def __getitem__(self, idx):
        I_smooth_1, I_sharp_1 = self._get_slice_pair(idx)
        I_smooth_2, I_sharp_2 = self._get_slice_pair(idx) 

        info = self.slice_data[idx]
        smooth_kernel_idx = torch.tensor(info['smooth_kernel_idx'], dtype=torch.long)
        sharp_kernel_idx  = torch.tensor(info['sharp_kernel_idx'],  dtype=torch.long)

        return I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2

    def __len__(self):
        return len(self.slice_data)
    