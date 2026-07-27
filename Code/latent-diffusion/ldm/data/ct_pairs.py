import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
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


class CTPairedBase(Dataset):
    #Slightly modified hte original dataset, return shape and normalization changed
    def __init__(self, root_dir, smooth_subdir="trainA", sharp_subdir="trainB",
                 min_slice_percentile=0.1, max_slice_percentile=0.9,
                 preload=True, seed=42, hu_min=-1000, hu_max=3000):
        self.smooth_dir = os.path.join(root_dir, smooth_subdir)
        self.sharp_dir = os.path.join(root_dir, sharp_subdir)
        self.min_percentile = min_slice_percentile
        self.max_percentile = max_slice_percentile
        self.preload = preload
        self.hu_min = hu_min
        self.hu_max = hu_max
        np.random.seed(seed)

        print("Finding volume pairs...")
        volume_pairs = self._find_volume_pairs()
        print(f"Found {len(volume_pairs)} volume pairs")

        self.volume_cache = self._preload_all_volumes(volume_pairs) if self.preload else {}
        if self.preload:
            print(f"Cached {len(self.volume_cache)} unique volumes")

        self.slice_data = self._build_slice_index(volume_pairs)
        print(f"Total slices: {len(self.slice_data)}")

        if self.preload and self.volume_cache:
            total_bytes = sum(v.nbytes for v in self.volume_cache.values())
            print(f"Memory usage: {total_bytes / (1024**3):.2f} GB")

    def _find_volume_pairs(self):
        smooth_files = sorted(f for f in os.listdir(self.smooth_dir) if f.endswith(('.nii', '.nii.gz')))
        sharp_files = sorted(f for f in os.listdir(self.sharp_dir) if f.endswith(('.nii', '.nii.gz')))

        sharp_dict = {
            (f.split("_filter_")[0] if "_filter_" in f else f.split(".")[0]): f
            for f in sharp_files
        }

        volume_pairs = []
        for sfile in smooth_files:
            base_id = sfile.split("_filter_")[0] if "_filter_" in sfile else sfile.split(".")[0]
            if base_id in sharp_dict:
                volume_pairs.append((sfile, sharp_dict[base_id]))
        return volume_pairs

    def _preload_all_volumes(self, volume_pairs):
        unique_paths = set()
        for sfile, shfile in volume_pairs:
            unique_paths.add(os.path.join(self.smooth_dir, sfile))
            unique_paths.add(os.path.join(self.sharp_dir, shfile))

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
            s_path = os.path.join(self.smooth_dir, sfile)
            sh_path = os.path.join(self.sharp_dir, shfile)

            try:
                smooth_kernel_str, smooth_kernel_idx = extract_kernel_from_filename(sfile)
                sharp_kernel_str, sharp_kernel_idx = extract_kernel_from_filename(shfile)
            except ValueError as e:
                print(f"Warning: {e} — skipping pair")
                continue

            if smooth_kernel_idx == -1 or sharp_kernel_idx == -1:
                print(f"Warning: unknown kernel in {sfile} or {shfile} — skipping")
                continue

            try:
                n_slices = (self.volume_cache[s_path].shape[2] if self.preload
                            else nib.load(s_path).shape[2])  # type: ignore

                start_idx = int(n_slices * self.min_percentile)
                end_idx = int(n_slices * self.max_percentile)

                for z_idx in range(start_idx, end_idx):
                    slice_data.append({
                        'smooth_path': s_path,
                        'sharp_path': sh_path,
                        'slice_idx': z_idx,
                        'smooth_kernel_str': smooth_kernel_str,
                        'smooth_kernel_idx': smooth_kernel_idx,
                        'sharp_kernel_str': sharp_kernel_str,
                        'sharp_kernel_idx': sharp_kernel_idx,
                    })
            except Exception as e:
                print(f"\nFailed to index {os.path.basename(s_path)}: {e}")
                continue

        return slice_data

    def _get_volume(self, path):
        if self.preload:
            return self.volume_cache[path]
        return nib.load(path).get_fdata().astype(np.float32)  # no caching at all

    def _normalize(self, img):
        # HU clip -> [-1, 1]
        img = np.clip(img, self.hu_min, self.hu_max)
        img = (img - self.hu_min) / (self.hu_max - self.hu_min)  # -> [0, 1]
        img = img * 2.0 - 1.0                                     # -> [-1, 1]
        return img.astype(np.float32)

    def _get_slice_pair(self, idx):
        info = self.slice_data[idx]
        vol_s = self._get_volume(info['smooth_path'])
        vol_h = self._get_volume(info['sharp_path'])

        img_s = self._normalize(vol_s[:, :, info['slice_idx']].copy())
        img_h = self._normalize(vol_h[:, :, info['slice_idx']].copy())

        # HWC, single channel — matches LatentDiffusion.get_input()'s
        # expected 'b h w c -> b c h w' rearrange
        img_s = img_s[..., None]
        img_h = img_h[..., None]
        return img_s, img_h

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        img_smooth, img_sharp = self._get_slice_pair(idx)
        return {
            "image": img_sharp,       # sharp CT = diffusion target
            "LR_image": img_smooth,   # smooth CT = conditioning input
        }


class CTPairedTrain(CTPairedBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CTPairedValidation(CTPairedBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class CTAutoencoderBase(Dataset):
    """Single-image dataset for stage-1 autoencoder training — trains on sharp CT only."""
    def __init__(self, root_dir, sharp_subdir="TrainB",preload=True, hu_min=-1000, hu_max=3000):
        self.sharp_dir = os.path.join(root_dir, sharp_subdir)
        self.preload = preload
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.volume_cache = {}
        self.slice_data = self._build_slice_index()

    def _build_slice_index(self):
        files = sorted(f for f in os.listdir(self.sharp_dir) if f.endswith(('.nii', '.nii.gz')))
        index = []
        for f in files:
            path = os.path.join(self.sharp_dir, f)
            vol = nib.load(path).get_fdata().astype(np.float32)  # type: ignore
            if self.preload:
                self.volume_cache[path] = vol
            n_slices = vol.shape[2]
            for z in range(int(n_slices * 0.1), int(n_slices * 0.9)):
                index.append((path, z))
        return index

    def _get_volume(self, path):
        if self.preload:
            return self.volume_cache[path]
        return nib.load(path).get_fdata().astype(np.float32)  # no caching at all

    def _normalize(self, img):
        img = np.clip(img, self.hu_min, self.hu_max)
        img = (img - self.hu_min) / (self.hu_max - self.hu_min) * 2.0 - 1.0
        return img.astype(np.float32)

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        path, z = self.slice_data[idx]
        img = self._normalize(self._get_volume(path)[:, :, z].copy())
        return {"image": img[..., None]}


class CTAutoencoderTrain(CTAutoencoderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CTAutoencoderValidation(CTAutoencoderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
