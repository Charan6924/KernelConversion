import torch
import nibabel as nib
from models.pix2pix import Pix2PixModel
from data.TestDataset import TestDataset
import matplotlib.pyplot as plt
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Pix2PixOptions:
    def __init__(self):
        self.name = 'pix2pix_sh2sm'
        self.isTrain = False
        self.checkpoints_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_pix2pix'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.verbose = False
        self.input_nc = 1        
        self.output_nc = 1   
        self.ngf = 64              
        self.ndf = 64              
        self.netG = 'unet_256'     # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        self.netD = 'basic'        # [basic | n_layers | pixel]
        self.n_layers_D = 3      
        self.norm = 'syncbatch'      # sync batch for ddp
        self.no_dropout = False    
        self.init_type = 'normal'  #[normal | xavier | kaiming | orthogonal]
        self.init_gain = 0.02  
        self.batch_size = 4        
        self.num_epochs = 200  

        self.direction = 'AtoB'    # Mapping direction ('AtoB' or 'BtoA')
        self.preprocess = 'none'  
        self.gan_mode = 'vanilla'  # [vanilla | lsgan | wgangp]
        self.lambda_L1 = 100.0  
        self.lr = 0.0002           
        self.beta1 = 0.5          

        self.lr_policy = 'linear'  # [linear | step | plateau | cosine]
        self.epoch_count = 1       
        self.n_epochs = 100        
        self.n_epochs_decay = 100  
        self.continue_train = False 
        
        self.load_iter = 0         
        self.epoch = 'latest'    

opt = Pix2PixOptions()
model = Pix2PixModel(opt)
model.setup(opt)
model.eval()

#load the checkpoint

print('Loaded model successfully')
data_root = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
dataset = TestDataset(root_dir=data_root, preload=True)
print('Loaded test dataset')

output_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_pix2pix"
os.makedirs(output_dir, exist_ok=True)

def extract_kernel_name(filename):
    if '_filter_' in filename:
        return filename.split('_filter_')[1].split('.')[0]
    return 'unknown'


def reconstruct_volume(sample, model, device, output_dir):
    data_smooth = sample['smooth_volume']
    data_sharp  = sample['sharp_volume']
    volume_id   = sample['volume_id']
    smooth_kernel = extract_kernel_name(sample['smooth_file'])
    sharp_kernel  = extract_kernel_name(sample['sharp_file'])
    num_slices = data_smooth.shape[2]

    #vol_generated_sharp  = np.zeros_like(data_smooth, dtype=np.float32)
    vol_generated_smooth = np.zeros_like(data_sharp,  dtype=np.float32)

    for k in range(num_slices):
        s_slice = data_smooth[:, :, k].copy()
        h_slice = data_sharp[:,  :, k].copy()

        s_slice = np.clip(s_slice, -1000, 3000)
        h_slice = np.clip(h_slice, -1000, 3000)

        s_slice_norm = (s_slice + 1000) / 4000 
        h_slice_norm = (h_slice + 1000) / 4000 

        I_smooth_tensor = torch.from_numpy(s_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        I_sharp_tensor  = torch.from_numpy(h_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            data = {'A': I_smooth_tensor, 'B': I_sharp_tensor, 'A_paths': '', 'B_paths': ''}
    
            model.set_input(data)
            model.forward()
            I_gen_smooth = model.fake_B

        #res_sharp  = I_gen_sharp.detach().cpu().numpy().squeeze()
        res_smooth = I_gen_smooth.detach().cpu().numpy().squeeze()

        #res_sharp  = res_sharp.clip(0, 1.0)
        res_smooth = res_smooth.clip(0, 1.0)

        #vol_generated_sharp[:,  :, k] = (res_sharp * 4000) - 1000
        vol_generated_smooth[:, :, k] = (res_smooth * 4000) - 1000

        #vol_generated_sharp[:,  :, k] = vol_generated_sharp[:,  :, k].clip(-1000, 3000)
        vol_generated_smooth[:, :, k] = vol_generated_smooth[:, :, k].clip(-1000, 3000)

    #nii_generated_sharp  = nib.Nifti1Image(vol_generated_sharp,  sample['sharp_affine'],  sample['sharp_header'])
    nii_generated_smooth = nib.Nifti1Image(vol_generated_smooth, sample['smooth_affine'], sample['smooth_header'])

    smooth_output_path  = os.path.join(output_dir, f'{volume_id}_{sharp_kernel}_to_{smooth_kernel}.nii.gz')
    #sharp_output_path = os.path.join(output_dir, f'{volume_id}_{smooth_kernel}_to_{sharp_kernel}.nii.gz')

    nib.save(nii_generated_smooth, smooth_output_path)

    print(f'Saved: {os.path.basename(smooth_output_path)}')


for idx in range(len(dataset)):
    print(f'\nProcessing volume {idx+1}/{len(dataset)}')
    sample = dataset[idx]
    reconstruct_volume(sample, model, device, output_dir)

print(f'\nReconstruction complete! All files saved to: {output_dir}')
