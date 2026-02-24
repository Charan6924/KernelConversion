import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MTFPSDDataset(Dataset):
    
    KERNEL_TO_IDX = {'B': 0, 'C': 1, 'CB': 2, 'D': 3, 'E': 4, 'YA': 5, 'YB': 6}
    
    def __init__(self, mtf_folder, psd_folder, transform=None, target_transform=None, return_paths=False, verbose=True):
        self.mtf_folder = Path(mtf_folder)
        self.psd_folder = Path(psd_folder)
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.verbose = verbose
        self.pairs = self._pair_files()

    def _extract_identifier(self, filename, file_type):
        if file_type == 'mtf':
            return filename.replace('_MTF_Results_mat.mat', '')
        elif file_type == 'psd':
            return filename.replace('_PSD.npy', '')
        return filename

    def _extract_kernel(self, identifier):
        # e.g. I10_Kernel_CB -> CB
        parts = identifier.split('_Kernel_')
        if len(parts) == 2:
            return parts[1]
        raise ValueError(f"Cannot extract kernel from identifier: {identifier}")

    def _pair_files(self):
        mtf_files = list(self.mtf_folder.glob('*_MTF_Results_mat.mat'))
        psd_files = list(self.psd_folder.glob('*_PSD.npy'))

        psd_dict = {self._extract_identifier(f.name, 'psd'): f for f in psd_files}

        pairs = []
        unmatched_mtf = []

        for mtf_file in mtf_files:
            identifier = self._extract_identifier(mtf_file.name, 'mtf')
            if identifier in psd_dict:
                kernel_str = self._extract_kernel(identifier)
                kernel_idx = self.KERNEL_TO_IDX.get(kernel_str, -1)
                if kernel_idx == -1:
                    if self.verbose:
                        print(f"Warning: Unknown kernel '{kernel_str}' in {identifier}")
                    continue
                pairs.append({
                    'identifier': identifier,
                    'mtf_path':   mtf_file,
                    'psd_path':   psd_dict[identifier],
                    'kernel_str': kernel_str,
                    'kernel_idx': kernel_idx,
                })
            else:
                unmatched_mtf.append(mtf_file.name)

        if self.verbose and unmatched_mtf:
            print(f"Warning: {len(unmatched_mtf)} MTF file(s) without matching PSD")

        pairs.sort(key=lambda x: x['identifier'])
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.pairs[idx]

        mtf_data = loadmat(str(pair['mtf_path']))
        results  = mtf_data['results']
        mtf_val  = results['mtfVal'][0, 0][0]
        mtf_val  = np.nan_to_num(mtf_val, nan=0.0)

        psd_data = np.load(str(pair['psd_path']))

        if self.transform:
            mtf_val = self.transform(mtf_val)
        if self.target_transform:
            psd_data = self.target_transform(psd_data)

        psd_data = np.array(psd_data, dtype=np.float32)
        mtf_val  = np.array(mtf_val,  dtype=np.float32).flatten()

        if psd_data.ndim == 2:
            input_profile = torch.from_numpy(psd_data).unsqueeze(0)
        elif psd_data.ndim == 3 and psd_data.shape[0] == 1:
            input_profile = torch.from_numpy(psd_data)
        else:
            input_profile = torch.from_numpy(psd_data.reshape(1, *psd_data.shape[-2:]))

        target_mtf  = torch.from_numpy(mtf_val)
        kernel_idx  = torch.tensor(pair['kernel_idx'], dtype=torch.long)

        return input_profile, target_mtf, kernel_idx

    def get_kernel_mtf_lookup(self, device='cpu'):
        """
        Returns a dict: kernel_idx (int) -> mean MTF tensor.
        Use this in training to get ground truth MTF for image batches.
        """
        accum = {}
        for pair in self.pairs:
            idx = pair['kernel_idx']
            mtf_data = loadmat(str(pair['mtf_path']))
            mtf_val  = mtf_data['results']['mtfVal'][0, 0][0]
            mtf_val  = np.nan_to_num(mtf_val, nan=0.0).astype(np.float32)
            if idx not in accum:
                accum[idx] = []
            accum[idx].append(mtf_val)

        return {
            k: torch.from_numpy(np.stack(v).mean(0)).to(device)
            for k, v in accum.items()
        }

    def get_sample_dict(self, idx):
        pair = self.pairs[idx]
        mtf_data = loadmat(str(pair['mtf_path']))
        psd_data = np.load(str(pair['psd_path']))
        sample = {
            'identifier': pair['identifier'],
            'kernel_str': pair['kernel_str'],
            'kernel_idx': pair['kernel_idx'],
            'mtf':        mtf_data,
            'psd':        psd_data,
        }
        if self.return_paths:
            sample['mtf_path'] = str(pair['mtf_path'])
            sample['psd_path'] = str(pair['psd_path'])
        return sample

    def get_identifiers(self):
        return [pair['identifier'] for pair in self.pairs]