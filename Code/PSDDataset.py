import os
import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import radial_average

class PSDDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sampling_strategy='uniform',
        slices_per_volume=10,
        min_slice_percentile=0.1, 
        max_slice_percentile=0.9,  
        window_level=40,         
        window_width=400,         
        use_ct_windowing=True,    
        cache_volumes=False,      
        augment=False,             
        seed=42
    ):
        self.smooth_dir = os.path.join(root_dir, "trainA")
        self.sharp_dir = os.path.join(root_dir, "trainB")
        self.sampling_strategy = sampling_strategy
        self.slices_per_volume = slices_per_volume
        self.min_percentile = min_slice_percentile
        self.max_percentile = max_slice_percentile
        self.window_level = window_level
        self.window_width = window_width
        self.use_ct_windowing = use_ct_windowing
        self.cache_volumes = cache_volumes
        self.augment = augment
        
        np.random.seed(seed)
        self.volume_cache = {} if cache_volumes else None
        smooth_files = sorted(os.listdir(self.smooth_dir))
        sharp_files = sorted(os.listdir(self.sharp_dir))
        
        volume_pairs = []
        for sfile in smooth_files:
            base_id = sfile.split("_filter_")[0] if "_filter_" in sfile else sfile.split(".")[0]
            for shfile in sharp_files:
                if base_id in shfile:
                    volume_pairs.append((sfile, shfile))
                    break
        
        print(f"Found {len(volume_pairs)} paired volumes")
        self.slice_data = []
        self._build_slice_index(volume_pairs)
        print(f"Total slices in dataset: {len(self.slice_data)}")
        print(f"Sampling strategy: {sampling_strategy}")
        if sampling_strategy != 'all':
            print(f"Slices per volume: {slices_per_volume}")
    
    def _build_slice_index(self, volume_pairs):
        for vol_idx, (sfile, shfile) in enumerate(volume_pairs):
            s_path = os.path.join(self.smooth_dir, sfile)
            sh_path = os.path.join(self.sharp_dir, shfile)
            try:
                smooth_img = nib.load(s_path)
                n_slices = smooth_img.shape[2]
            except Exception as e:
                print(f"Warning: Could not load {sfile}: {e}")
                continue
            start_idx = int(n_slices * self.min_percentile)
            end_idx = int(n_slices * self.max_percentile)
            valid_range = end_idx - start_idx
            
            if valid_range <= 0:
                print(f"Warning: No valid slices for {sfile}")
                continue
            
            if self.sampling_strategy == 'all':
                slice_indices = list(range(start_idx, end_idx))
            
            elif self.sampling_strategy == 'uniform':
                if valid_range >= self.slices_per_volume:
                    slice_indices = np.linspace(
                        start_idx, end_idx - 1, 
                        self.slices_per_volume, 
                        dtype=int
                    ).tolist()
                else:
                    slice_indices = list(range(start_idx, end_idx))
            
            elif self.sampling_strategy == 'random':
                n_samples = min(self.slices_per_volume, valid_range)
                slice_indices = np.random.choice(
                    range(start_idx, end_idx),
                    size=n_samples,
                    replace=False
                ).tolist()
            
            elif self.sampling_strategy == 'adaptive':
                data = nib.load(s_path).get_fdata()
                slice_variances = []
                for z in range(start_idx, end_idx):
                    slice_var = np.var(data[:, :, z])
                    slice_variances.append((z, slice_var))
                
                slice_variances.sort(key=lambda x: x[1], reverse=True)
                n_samples = min(self.slices_per_volume, len(slice_variances))
                slice_indices = [z for z, _ in slice_variances[:n_samples]]
                slice_indices.sort()
            
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            for z_idx in slice_indices:
                self.slice_data.append({
                    'smooth_path': s_path,
                    'sharp_path': sh_path,
                    'slice_idx': z_idx,
                    'volume_idx': vol_idx
                })
    
    def __len__(self):
        return len(self.slice_data)
    
    def _load_volume(self, path):
        if self.volume_cache is not None:
            if path not in self.volume_cache:
                self.volume_cache[path] = nib.load(path).get_fdata()
            return self.volume_cache[path]
        else:
            return nib.load(path).get_fdata()
    
    def apply_ct_windowing(self, img, level=None, width=None):
        level = level or self.window_level
        width = width or self.window_width
        
        img_min = level - width / 2
        img_max = level + width / 2
        
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min + 1e-8)
        
        return img
    
    def apply_minmax_normalization(self, img):
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)
    
    def augment_slice(self, smooth_slice, sharp_slice):
        if np.random.rand() > 0.5:
            smooth_slice = np.fliplr(smooth_slice)
            sharp_slice = np.fliplr(sharp_slice)
        
        if np.random.rand() > 0.5:
            smooth_slice = np.flipud(smooth_slice)
            sharp_slice = np.flipud(sharp_slice)
        
        k = np.random.randint(0, 4) 
        if k > 0:
            smooth_slice = np.rot90(smooth_slice, k)
            sharp_slice = np.rot90(sharp_slice, k)
        
        return smooth_slice, sharp_slice
    
    def __getitem__(self, idx):
        sample_info = self.slice_data[idx]
        I_smooth = self._load_volume(sample_info['smooth_path'])
        I_sharp = self._load_volume(sample_info['sharp_path'])
        z_idx = sample_info['slice_idx']
        I_smooth_slice = I_smooth[:, :, z_idx].copy()
        I_sharp_slice = I_sharp[:, :, z_idx].copy()

        if self.augment:
            I_smooth_slice, I_sharp_slice = self.augment_slice(I_smooth_slice, I_sharp_slice)

        if self.use_ct_windowing:
            I_smooth_slice = self.apply_ct_windowing(I_smooth_slice)
            I_sharp_slice = self.apply_ct_windowing(I_sharp_slice)
        else:
            I_smooth_slice = self.apply_minmax_normalization(I_smooth_slice)
            I_sharp_slice = self.apply_minmax_normalization(I_sharp_slice)
        psd_smooth = radial_average(I_smooth_slice)
        psd_sharp = radial_average(I_sharp_slice)
        I_smooth_slice = torch.tensor(I_smooth_slice, dtype=torch.float32).unsqueeze(0)
        I_sharp_slice = torch.tensor(I_sharp_slice, dtype=torch.float32).unsqueeze(0)
        psd_smooth = torch.tensor(psd_smooth, dtype=torch.float32).unsqueeze(0)
        psd_sharp = torch.tensor(psd_sharp, dtype=torch.float32).unsqueeze(0)
        
        return I_smooth_slice, I_sharp_slice, psd_smooth, psd_sharp
    
    def get_volume_info(self):
        volume_info = {}
        for sample in self.slice_data:
            vol_idx = sample['volume_idx']
            if vol_idx not in volume_info:
                volume_info[vol_idx] = {
                    'smooth_path': sample['smooth_path'],
                    'sharp_path': sample['sharp_path'],
                    'n_slices': 0
                }
            volume_info[vol_idx]['n_slices'] += 1
        
        return volume_info
