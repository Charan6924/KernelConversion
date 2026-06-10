from collections import defaultdict
import re
import os
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def inspect_series(root_dir):
    series_info = {}
    for series_dir in sorted(Path(root_dir).iterdir()):
        if not series_dir.is_dir():
            continue
        try:
            dicoms = sorted(series_dir.glob("I*"))
            if not dicoms:
                print(f"{series_dir.name} | No I* files found")
                continue
            ds = pydicom.dcmread(dicoms[0], stop_before_pixels=True)
            kernel = getattr(ds, 'ConvolutionKernel', 'N/A')
            desc   = getattr(ds, 'SeriesDescription', '')
            num    = getattr(ds, 'SeriesNumber', -1)
            series_info[series_dir.name] = {
                "path":        series_dir,
                "files":       dicoms,
                "kernel":      kernel,
                "description": desc,
                "series_num":  num,
            }
        except pydicom.errors.InvalidDicomError:
            print(f"{series_dir.name} | ERROR: Not a valid DICOM file")
        except PermissionError:
            print(f"{series_dir.name} | ERROR: Permission denied")
        except Exception as e:
            print(f"{series_dir.name} | ERROR: {type(e).__name__}: {e}")
    return series_info


def idose_level(desc):
    desc = desc.strip().lower()
    if "idose" in desc:
        m = re.search(r'\((\d)\)', desc)
        return f"idose_{m.group(1)}" if m else "idose_?"
    return "fbp"

def dicom_to_normalized(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    arr = arr * slope + intercept 
    # arr = np.clip(arr, -1000, 3000)
    # arr = (arr + 1000) / 4000      
    return arr

def pair_series(series_info, smooth_kernels=("B", "C"), sharp_kernels=("YA", "YB")):
    buckets = {}
    for _ , meta in series_info.items():
        if meta["kernel"] == "N/A":
            continue
        key = (meta["kernel"], idose_level(meta["description"]))
        buckets[key] = meta

    pairs = []
    for s_kern in smooth_kernels:
        for sh_kern in sharp_kernels:
            s_keys  = {k[1] for k in buckets if k[0] == s_kern}
            sh_keys = {k[1] for k in buckets if k[0] == sh_kern}
            shared  = s_keys & sh_keys
            for level in sorted(shared):
                pairs.append({
                    "smooth":      buckets[(s_kern, level)],
                    "sharp":       buckets[(sh_kern, level)],
                    "kernel_pair": (s_kern, sh_kern),
                    "idose":       level,
                })
    return pairs


def load_volume(series_meta):
    """Load all slices in a series into a (Z, H, W) float32 array."""
    slices = []
    for f in sorted(series_meta["files"]):
        ds = pydicom.dcmread(f)
        slices.append(ds.pixel_array.astype(np.float32))
    return np.stack(slices, axis=0)  # (Z, H, W)

class KernelPairDataset(Dataset):
    def __init__(self, root_dir, smooth_kernels=("B", "C"), sharp_kernels=("YA", "YB")):
        series_info = inspect_series(root_dir)
        pairs = pair_series(series_info, smooth_kernels, sharp_kernels)

        self.samples = []
        for pair in pairs:
            self.samples.append({
                "smooth_paths": sorted(pair["smooth"]["files"]),
                "sharp_paths":  sorted(pair["sharp"]["files"]),
                "kernel_pair":  pair["kernel_pair"],
                "idose":        pair["idose"],
            })

        print(f"Dataset ready: {len(self.samples)} volume pairs")

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, paths) -> torch.Tensor:
        slices = [dicom_to_normalized(pydicom.dcmread(str(p))) for p in paths]
        vol = np.stack(slices, axis=0).astype(np.float32)  
        return torch.from_numpy(vol)   # ( D, H, W)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "smooth": self._load_volume(s["smooth_paths"]),  
            "sharp": self._load_volume(s["sharp_paths"]),   
            "kernel_pair": s["kernel_pair"],                    
            "idose": s["idose"],                           
        }


