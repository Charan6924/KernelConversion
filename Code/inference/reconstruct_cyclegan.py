import torch
import nibabel as nib
import numpy as np
import os 
from models.cycle_gan_model import CycleGANModel
from data.TestDataset import TestDataset

class CycleGANOptions:
    def __init__(self):
        self.dataroot = 'placeholder'
        self.name = 'Cycle-GAN'
        self.easy_label = 'Cycle-GAN'
        self.gpu_ids = [0]
        self.checkpoints_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_cyclegan'
        self.model = 'cycle_gan'
        self.input_nc = 1
        self.output_nc = 1
        self.preprocess = 'none'
        self.ngf = 64
        self.ndf = 64
        self.netG = 'resnet_9blocks'
        self.netD = 'basic'
        self.n_layers_D = 3
        self.normG = 'instance'
        self.normD = 'instance'
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_dropout = True
        self.no_antialias = False
        self.no_antialias_up = False
        self.stylegan2_G_num_downsampling = 1
        self.dataset_mode = 'unaligned'
        self.direction = 'AtoB'
        self.serial_batches = False
        self.num_threads = 4
        self.batch_size = 1
        self.max_dataset_size = float('inf')
        self.no_flip = False
        self.display_winsize = 256
        self.random_scale_max = 3.0
        self.epoch = 'latest'
        self.verbose = False
        self.suffix = ''
        self.isTrain = False
        self.phase = 'test'
        self.amp = False
        self.display_freq = 400
        self.display_ncols = 4
        self.display_id = None
        self.display_server = 'http://localhost'
        self.display_env = 'main'
        self.display_port = 8097
        self.update_html_freq = 1000
        self.print_freq = 100
        self.no_html = False
        self.save_latest_freq = 5000
        self.save_epoch_freq = 5
        self.evaluation_freq = 5000
        self.save_by_iter = False
        self.continue_train = False
        self.epoch_count = 1
        self.pretrained_name = None
        self.n_epochs = 100
        self.n_epochs_decay = 100
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = 0.0002
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.gan_mode = 'lsgan'
        self.pool_size = 50
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_identity = 0.5

opt = CycleGANOptions()
model = CycleGANModel(opt)
model.setup(opt)
model.eval()

print('Loaded model successfully')
data_root = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
dataset = TestDataset(root_dir=data_root, preload=True)
print('Loaded test dataset')

output_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_cyclegan"
os.makedirs(output_dir, exist_ok=True)

def extract_kernel_name(filename):
    if '_filter_' in filename:
        return filename.split('_filter_')[1].split('.')[0]
    return 'unknown'


def reconstruct_volume(sample, model, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'testA_fake'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'testB_fake'), exist_ok=True)
    data_smooth = sample['smooth_volume']
    data_sharp  = sample['sharp_volume']
    volume_id   = sample['volume_id']
    smooth_kernel = extract_kernel_name(sample['smooth_file'])
    sharp_kernel  = extract_kernel_name(sample['sharp_file'])
    num_slices = data_smooth.shape[2]

    vol_generated_sharp  = np.zeros_like(data_smooth, dtype=np.float32)
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
            I_gen_smooth = model.fake_A
            I_gen_sharp = model.fake_B

        res_sharp  = I_gen_sharp.detach().cpu().numpy().squeeze()
        res_smooth = I_gen_smooth.detach().cpu().numpy().squeeze()

        res_sharp  = res_sharp.clip(0, 1.0)   
        res_smooth = res_smooth.clip(0, 1.0)

        vol_generated_sharp[:,  :, k] = (res_sharp  * 4000) - 1000
        vol_generated_smooth[:, :, k] = (res_smooth * 4000) - 1000
        vol_generated_sharp[:,  :, k] = vol_generated_sharp[:,  :, k].clip(-1000, 3000)
        vol_generated_smooth[:, :, k] = vol_generated_smooth[:, :, k].clip(-1000, 3000)

    nii_generated_sharp  = nib.Nifti1Image(vol_generated_sharp,  sample['sharp_affine'],  sample['sharp_header'])
    nii_generated_smooth = nib.Nifti1Image(vol_generated_smooth, sample['smooth_affine'], sample['smooth_header'])

    smooth_output_path  = os.path.join(output_dir,'testA_fake', f'{volume_id}_{sharp_kernel}_to_{smooth_kernel}.nii.gz')
    sharp_output_path = os.path.join(output_dir, 'testB_fake',f'{volume_id}_{smooth_kernel}_to_{sharp_kernel}.nii.gz')
    
    nib.save(nii_generated_smooth, smooth_output_path)
    nib.save(nii_generated_sharp, sharp_output_path)


    print(f'Saved: {os.path.basename(smooth_output_path)}, {os.path.basename(sharp_output_path)}')


for idx in range(len(dataset)):
    print(f'\nProcessing volume {idx+1}/{len(dataset)}')
    sample = dataset[idx]
    device = 'cuda'
    reconstruct_volume(sample, model, device, output_dir)

print(f'\nReconstruction complete! All files saved to: {output_dir}')
