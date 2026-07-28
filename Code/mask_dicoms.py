import pydicom
import tifffile
import numpy as np

dcm_file = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2020/I20"
ds = pydicom.dcmread(dcm_file)
arr = ds.pixel_array.astype(np.float32)
slope = float(getattr(ds, 'RescaleSlope', 1))
intercept = float(getattr(ds, 'RescaleIntercept', 0))
arr = arr * slope + intercept
arr = np.clip(arr, -1000, 3000)
arr = (arr + 1000) / 4000

mask_paths = ["/home/cxv166/PhantomTesting/Code/masks/Mask_LD.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_RD.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_LU.tif",
              "/home/cxv166/PhantomTesting/Code/masks/Mask_RU.tif"]

mask_names = ['LD','RD','LU','RU']

masks = [tifffile.imread(p) for p in mask_paths]

results = {}
for name, mask in zip(mask_paths, masks):
    binary_mask = mask > 0

    if binary_mask.shape != arr.shape:
        raise ValueError(f"Shape mismatch, mask shape is {binary_mask.shape}, img shape is {arr.shape}")

    region_values = arr[binary_mask]
    mean_val = region_values.mean()
    std_val = region_values.std()
    mean_hu = mean_val * 4000 - 1000
    std_hu = std_val * 4000 

    print(mean_hu,std_hu)
