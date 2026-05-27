import torch
import torch.nn as nn
import torch.nn.functional as F
# from data.CTA3DDataset import CTA3DDataset
# from models.Unet3D import UNet3D as Unet

from torch.utils.data import DataLoader
# from utils.tools import all_reduce_tensor
from torch import nn
# from models import criterions
import argparse
import os
import random
import logging
import numpy as np
import time
from torch.nn import init
import torch
import torch.backends.cudnn as cudnn
import torch.optim

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from collections.abc import Iterable
import nibabel as nib
from torchvision.transforms import transforms
from PIL import Image
import functools
from packaging import version
# import matplotlib.pyplot as plt
from sklearn import datasets
import cv2
from models import networks
from scipy.ndimage import zoom

class ClipNormalize(object):
    def __call__(self, sample):
        image = sample

        Max = 3000.0
        Min = -1000.0
        image = np.clip(image, Min, Max)
        image = (image - Min) / (Max - Min)

        return image
def normalize_tensor(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor        

def resize_slice(slice_data, target_size=(256, 256)):
    zoom_factors = (target_size[0] / slice_data.shape[0], target_size[1] / slice_data.shape[1])
    resized_slice = zoom(slice_data, zoom_factors, order=1)  # Using bilinear interpolation
    return resized_slice

def compute_psd(volume):
    psd = []
    # ft = []
    for slice_idx in range(volume.shape[2]):
        slice_data = volume[:, :, slice_idx]
        slice_ft = torch.fft.fftshift(torch.fft.fft2(torch.tensor(slice_data, dtype=torch.float32)))
        slice_psd = torch.abs(slice_ft) ** 2
        slice_psd = torch.log(slice_psd+1)
        slice_psd = normalize_tensor(slice_psd)
        psd.append(slice_psd)
    psd = torch.stack(psd, dim=2)
    return psd

def construct_full_kernel(half_kernel):
    B, C, H, W = half_kernel.shape
    full_kernel = torch.zeros(B, C, H, 2 * W).to(half_kernel.device)
    # Top Half
    full_kernel[:, :, :H, :W] = half_kernel
    # Bottom Half
    full_kernel[:, :, :H, W:] = torch.flip(half_kernel, [2,3])
    return full_kernel 

def inference_kernel(cur_volume, network):
    # cur_volume_resized = np.stack([resize_slice(cur_volume[:, :, i]) for i in range(cur_volume.shape[2])], axis=2)
    cur_volume_resized = cur_volume
    psd = compute_psd(cur_volume_resized)
    
    psd = psd.unsqueeze(0).permute(3, 0, 1, 2)
    # print(f'psd shape: {psd.shape}' )
    meanlist = 0.5*np.ones(cur_volume_resized.shape[2])
    stdlist = 0.5*np.ones(cur_volume_resized.shape[2])
    transform_list = []
    transform_list += [ClipNormalize(), transforms.ToTensor()]
    transform_list += [transforms.Normalize(meanlist, stdlist)]
    t = transforms.Compose(transform_list)
    volume_tensor = t(cur_volume_resized).unsqueeze(0).float().permute(1, 0, 2, 3)
    # print(f'volume_tensor shape: {volume_tensor.shape}' )
    ft_image = fourier_transform(volume_tensor)
    return network(psd), ft_image

def fourier_transform(image):
    # print(f'image_ft shape: {image.shape}')
    image_ft = torch.fft.fftshift(torch.fft.fft2(image,dim=(-2, -1)))
    return image_ft 
def convert_ft(k1, k2):
    # return torch.div(k1, k2 + 1e-10, rounding_mode='trunc')
    return k1 / (k2 + 1e-10)
def inverse_fourier_transform(image):
    return torch.fft.ifft2(torch.fft.ifftshift(image)).real

if __name__ == '__main__':
    netG = networks.define_G(1, 1, 64, 'resnet_KC', 'instance', not True, 'xavier', 0.02, False, False, [])
    pth_path = r"D:\Charan work file\For Charan\kernelConversion\code\latest_net_G.pth" # dir in HPC, I believe lab members can access it.
    state_dict = torch.load(pth_path, map_location=torch.device('cpu')) 
    netG.load_state_dict(state_dict)
    netG.eval()

    processed_dir_smooth = r"D:\Charan work file\PhantomTesting\testA"
    processed_dir_sharp = r"D:\Charan work file\PhantomTesting\testB"
    sav_smooth_dir = r"D:\Charan work file\PhantomTesting\testA_fake"
    sav_sharp_dir = r"D:\Charan work file\PhantomTesting\testB_fake"

    smooth_paths = sorted([os.path.join(processed_dir_smooth, name) for name in os.listdir(processed_dir_smooth)])
    sav_smooth_pathes = sorted([os.path.join(sav_smooth_dir, name) for name in os.listdir(processed_dir_smooth)])

    sharp_paths = sorted([os.path.join(processed_dir_sharp, name) for name in os.listdir(processed_dir_sharp)])
    sav_sharp_paths = sorted([os.path.join(sav_sharp_dir, name) for name in os.listdir(processed_dir_sharp)])

    paths = zip(smooth_paths, sharp_paths, sav_smooth_pathes, sav_sharp_paths)

    count = 0 
    for sm_name, sh_name, sav_sm_name, sav_sh_name in paths:
            # print(sav_name)
            # print(os.path.isfile(data_name))
        print(f'processing case # {count}')    
        print(f'processing: {sav_sm_name}')
        print(f'processing: {sav_sh_name}')
        count = count + 1
        smooth =  nib.load(sm_name).get_fdata()
        sharp = nib.load(sh_name).get_fdata()
        # sx1, sy1, sz1 = smooth.shape
        # sx2, sy2, sz2 = sharp.shape
        # print(f'current processing: {sav_sm_name}')
        # if sz1 != sz2:
        #     count = count + 1
        #     print(f'processing case # {count}')    
        #     print(f'processing: {sav_sm_name}')
        #     print(f'processing: {sav_sh_name}')


        sm_kernel,ft_smooth = inference_kernel(smooth, netG)
        sh_kernel,ft_sharp = inference_kernel(sharp, netG)

        sm_kernel = construct_full_kernel(sm_kernel)
        sh_kernel = construct_full_kernel(sh_kernel)

        k_smooth_complex = torch.complex(sm_kernel[:,0,:,:], sm_kernel[:,1,:,:]).unsqueeze(1)
        k_sharp_complex = torch.complex(sh_kernel[:,0,:,:], sh_kernel[:,1,:,:]).unsqueeze(1)

        filter_smooth2sharp = convert_ft(k_sharp_complex, k_smooth_complex)
        filter_sharp2smooth = convert_ft(k_smooth_complex, k_sharp_complex)

        fake_sharp_ft = ft_smooth*filter_smooth2sharp
        fake_smooth_ft = ft_sharp*filter_sharp2smooth

        fake_smooth = inverse_fourier_transform(fake_smooth_ft)
        fake_sharp = inverse_fourier_transform(fake_sharp_ft)
        
        fake_smooth = fake_smooth.clamp(-1.0, 1.0).cpu().squeeze().permute(1,2,0).float().detach().numpy()
        fake_sharp = fake_sharp.clamp(-1.0, 1.0).cpu().squeeze().permute(1,2,0).float().detach().numpy()

        min_old, max_old =  -1.0, 1.0
        min_new, max_new = -1000, 3000.0

        fake_smooth =  min_new + ((fake_smooth - min_old)*(max_new - min_new) / (max_old - min_old))
        fake_sharp =  min_new + ((fake_sharp - min_old)*(max_new - min_new) / (max_old - min_old))

        nib.save(nib.Nifti1Image(fake_smooth, np.eye(4)), sav_sm_name)
        nib.save(nib.Nifti1Image(fake_sharp, np.eye(4)), sav_sh_name)



