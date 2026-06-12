import numpy as np
import nibabel as nib
import torch
import os
import matplotlib
from torch.utils.data import DataLoader
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.utils import compute_psd, spline_to_kernel, generate_images
from data.Dataset import MTFPSDDataset
from data.PSDDataset import PSDDataset
from models.KernelEstimator import KernelEstimator
from utils.utils import compute_fft, spline_to_kernel
from scipy.interpolate import CubicSpline

'''
Reconstructing patient volumes with ground truth phantom measurements
'''

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = KernelEstimator()
# checkpoint = torch.load("/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_kernel256/checkpoints/best_checkpoint.pth", map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)
# model.eval()
# print('Loaded model successfully')

mtf_e = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_CB_MTF_Results_mat.mat')
mtf_d = loadmat('/home/cxv166/PhantomTesting/MTF_Results_Output/I20_Kernel_YA_MTF_Results_mat.mat')
