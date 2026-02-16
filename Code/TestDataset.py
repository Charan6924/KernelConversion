import os
import nibabel as nib
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import warnings

class TestDataset(Dataset):
    def __init__(self, root_dir,
                 preload=True, seed=42):
        self.smooth_dir = os.path.join(root_dir, "testA")
        self.sharp_dir = os.path.join(root_dir, "testB")
        self.preload = preload
        np.random.seed(seed)
        #woof
        
        print("Finding volume pairs...")
        self.volume_pairs = self._find_volume_pairs()
        print(f"Found {len(self.volume_pairs)} volume pairs")
        
        if self.preload:
            self.volume_cache = self._preload_all_volumes(self.volume_pairs)
            print(f"Cached {len(self.volume_cache)} unique volumes")
        else:
            self.volume_cache = {}
            print("Using lazy loading - first epoch will be slower")
        
    def _find_volume_pairs(self):
        smooth_files = sorted([f for f in os.listdir(self.smooth_dir) 
                              if f.endswith(('.nii', '.nii.gz'))])
        sharp_files = sorted([f for f in os.listdir(self.sharp_dir) 
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
            unique_paths.add(os.path.join(self.sharp_dir, shfile))
        
        cache = {}
        for path in tqdm(sorted(unique_paths), desc="Loading volumes"):
            try:
                nii_img = nib.load(path)
                cache[path] = {
                    'data': nii_img.get_fdata(),
                    'affine': nii_img.affine,
                    'header': nii_img.header
                }
            except Exception as e:
                print(f"\nFailed to load {os.path.basename(path)}: {e}")
        
        return cache
    
    def __len__(self):
        return len(self.volume_pairs)
    
    def __getitem__(self, idx):
        smooth_file, sharp_file = self.volume_pairs[idx]
        
        smooth_path = os.path.join(self.smooth_dir, smooth_file)
        sharp_path = os.path.join(self.sharp_dir, sharp_file)
        
        if self.preload:
            smooth_data = self.volume_cache[smooth_path]['data']
            sharp_data = self.volume_cache[sharp_path]['data']
            smooth_affine = self.volume_cache[smooth_path]['affine']
            sharp_affine = self.volume_cache[sharp_path]['affine']
            smooth_header = self.volume_cache[smooth_path]['header']
            sharp_header = self.volume_cache[sharp_path]['header']
        else:
            smooth_nii = nib.load(smooth_path)
            sharp_nii = nib.load(sharp_path)
            smooth_data = smooth_nii.get_fdata()
            sharp_data = sharp_nii.get_fdata()
            smooth_affine = smooth_nii.affine
            sharp_affine = sharp_nii.affine
            smooth_header = smooth_nii.header
            sharp_header = sharp_nii.header
        
        volume_id = smooth_file.split("_filter_")[0] if "_filter_" in smooth_file else smooth_file.split(".")[0]
        
        return {
            'smooth_volume': smooth_data,
            'sharp_volume': sharp_data,
            'volume_id': volume_id,
            'smooth_affine': smooth_affine,
            'sharp_affine': sharp_affine,
            'smooth_header': smooth_header,
            'sharp_header': sharp_header,
            'smooth_file': smooth_file,
            'sharp_file': sharp_file
        }


root_dir = r"D:\Charan work file\KernelEstimator\Data_Root"
test_dataset = TestDataset(root_dir=root_dir)

print(f"\nDataset length: {len(test_dataset)}")

sample = test_dataset[0]
print(f"\nFirst sample volume ID: {sample['volume_id']}")
print(f"Smooth volume shape: {sample['smooth_volume'].shape}")
print(f"Sharp volume shape: {sample['sharp_volume'].shape}")
print(f"Smooth file: {sample['smooth_file']}")
print(f"Sharp file: {sample['sharp_file']}")
print(f"Affine shape: {sample['smooth_affine'].shape}")

print("\nIterating through first 3 volumes:")
for i in range(min(3, len(test_dataset))):
    vol = test_dataset[i]
    print(f"Volume {i}: {vol['volume_id']}, Shape: {vol['smooth_volume'].shape}")