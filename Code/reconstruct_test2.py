'''using the trained model to reconstruct phantom volumes'''

import torch
import pydicom
import numpy as np
from models.KernelEstimator import KernelEstimator
import os
from utils.utils import compute_psd,compute_fft, generate_images, spline_to_kernel

ds1 = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2030/I20')
pixel_array1 = ds1.pixel_array
ds2 = pydicom.dcmread('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2050/I20')
pixel_array2 = ds2.pixel_array

model = KernelEstimator()
model.to('cuda')
checkpoint = torch.load('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_kernel256/checkpoints/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(ds1.RescaleSlope, ds1.RescaleIntercept)

def dicom_to_normalized(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    arr = arr * slope + intercept 
    arr = np.clip(arr, -1000, 3000)
    arr = (arr + 1000) / 4000      
    return arr

pixel_array1 = torch.from_numpy(dicom_to_normalized(ds1)).unsqueeze(0).unsqueeze(0)
pixel_array2 = torch.from_numpy(dicom_to_normalized(ds2)).unsqueeze(0).unsqueeze(0)

print(pixel_array1.min(),pixel_array1.max())

psd_1 = compute_psd(pixel_array1, device = 'cuda').to('cuda')
psd_2 = compute_psd(pixel_array2, device = 'cuda').to('cuda')

print(psd_1.shape)

with torch.no_grad():
    mtf_1 = model(psd_1)
    mtf_2 = model(psd_2)

kernel_1, kernel_2 = spline_to_kernel(mtf_1,mtf_2)

filter1to2 = kernel_2/(kernel_1 + 1e-10)
filter2to1 = kernel_1/(kernel_2 + 1e-10)

Image_generated1, Image_generated2 = generate_images(pixel_array1,pixel_array2,filter1to2,filter2to1)

def save_to_dicom(original_ds, reconstructed_array, output_path):
    ds = original_ds  
    arr = reconstructed_array.squeeze().cpu().numpy() 
    arr = (arr * 4000) - 1000  
    arr = np.clip(arr, -1000, 3000)
    arr = (arr - float(ds.RescaleIntercept)) / float(ds.RescaleSlope) 
    arr = np.round(arr).astype(np.uint16)
    
    ds.PixelData = arr.tobytes()
    ds.save_as(output_path)

os.makedirs('/home/cxv166/PhantomTesting/Code/S2030_recon', exist_ok=True)
os.makedirs('/home/cxv166/PhantomTesting/Code/S2050_recon', exist_ok=True)

save_to_dicom(ds1, Image_generated1, '/home/cxv166/PhantomTesting/Code/S2030_recon/I20')
save_to_dicom(ds2, Image_generated2, '/home/cxv166/PhantomTesting/Code/S2050_recon/I20')
