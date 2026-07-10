import torch
import nibabel as nib
import numpy as np
import os
from data.TestDataset import TestDataset
from models.cut_model import CUTModel

class CUTSharp2SmoothOptions:
    def __init__(self):
        self.dataroot = 'mnt'
        self.name = 'CUTModelSharp2Smooth'
        self.easy_label = 'cut_train'
        self.gpu_ids = [0] 
        self.checkpoints_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_cut'
        self.preprocess = 'none'
        self.model = 'cut'
        self.input_nc = 1
        self.output_nc = 1
        self.ngf = 64
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'resnet_9blocks'
        self.n_layers_D = 3
        self.normG = 'instance'
        self.normD = 'instance'
        self.init_type = 'xavier'
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
        self.n_epochs = 200
        self.n_epochs_decay = 200
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = 0.0002
        self.gan_mode = 'lsgan'
        self.pool_size = 0  
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.CUT_mode = 'CUT'
        self.lambda_GAN = 1.0
        self.lambda_smooth = 0.0
        self.lambda_NCE = 1.0
        self.nce_idt = True
        self.nce_layers = '0,4,8,12,16'
        self.nce_includes_all_negatives_from_minibatch = False
        self.netF = 'mlp_sample'
        self.netF_nc = 256
        self.nce_T = 0.07
        self.num_patches = 256
        self.flip_equivariance = False 
        self.device = 'cuda:0'
        self.lambda_spa_unsup_A = 1.0
        self.lambda_kl = 1.0
        self.unsup_idt_spa = True
        self.motion_level = 8.0
        self.shift_level = 10.0
        self.scale_level = 0.0
        self.noise_level = 0.001


class CUTSmooth2SharpOptions(CUTSharp2SmoothOptions):
    def __init__(self):
        super().__init__()
        self.name = 'CUTModel'  # smooth to sharp checkpoint


opt_s2h = CUTSharp2SmoothOptions()  # sharp to smooth
model_s2h = CUTModel(opt_s2h)
model_s2h.setup(opt_s2h)
model_s2h.eval()
print('Loaded Sharp2Smooth model')

opt_h2s = CUTSmooth2SharpOptions()  # smooth to sharp
model_h2s = CUTModel(opt_h2s)
model_h2s.setup(opt_h2s)
model_h2s.eval()
print('Loaded Smooth2Sharp model')

data_root = "/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
dataset = TestDataset(root_dir=data_root, preload=True)
print('Loaded test dataset')

output_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_cut"
os.makedirs(output_dir, exist_ok=True)


def extract_kernel_name(filename):
    if '_filter_' in filename:
        return filename.split('_filter_')[1].split('.')[0]
    return 'unknown'


def reconstruct_volume(sample, model_s2h, model_h2s, device, output_dir):
    data_smooth = sample['smooth_volume']
    data_sharp  = sample['sharp_volume']
    volume_id   = sample['volume_id']
    smooth_kernel = extract_kernel_name(sample['smooth_file'])
    sharp_kernel  = extract_kernel_name(sample['sharp_file'])
    num_slices = data_smooth.shape[2]

    os.makedirs(os.path.join(output_dir, 'testA_fake'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'testB_fake'), exist_ok=True)

    vol_generated_smooth = np.zeros_like(data_smooth, dtype=np.float32)
    vol_generated_sharp  = np.zeros_like(data_sharp,  dtype=np.float32)

    for k in range(num_slices):
        s_slice = np.clip(data_smooth[:, :, k].copy(), -1000, 3000)
        h_slice = np.clip(data_sharp[:,  :, k].copy(), -1000, 3000)

        s_slice_norm = (s_slice + 1000) / 4000  # [0, 1]
        h_slice_norm = (h_slice + 1000) / 4000  # [0, 1]

        I_smooth_tensor = torch.from_numpy(s_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        I_sharp_tensor  = torch.from_numpy(h_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            data_s2h = {'A': I_sharp_tensor, 'B': I_smooth_tensor, 'A_paths': '', 'B_paths': ''}
            model_s2h.set_input(data_s2h)
            model_s2h.forward()
            res_smooth = model_s2h.fake_B.detach().cpu().numpy().squeeze()

            data_h2s = {'A': I_smooth_tensor, 'B': I_sharp_tensor, 'A_paths': '', 'B_paths': ''}
            model_h2s.set_input(data_h2s)
            model_h2s.forward()
            res_sharp = model_h2s.fake_B.detach().cpu().numpy().squeeze()

        res_smooth = res_smooth.clip(0, 1.0)
        res_sharp  = res_sharp.clip(0, 1.0)

        vol_generated_smooth[:, :, k] = ((res_smooth * 4000) - 1000).clip(-1000, 3000)
        vol_generated_sharp[:,  :, k] = ((res_sharp  * 4000) - 1000).clip(-1000, 3000)

    nii_smooth = nib.Nifti1Image(vol_generated_smooth, sample['smooth_affine'], sample['smooth_header'])
    nii_sharp  = nib.Nifti1Image(vol_generated_sharp,  sample['sharp_affine'],  sample['sharp_header'])

    smooth_output_path = os.path.join(output_dir, 'testA_fake', f'{volume_id}_{sharp_kernel}_to_{smooth_kernel}.nii.gz')
    sharp_output_path  = os.path.join(output_dir, 'testB_fake', f'{volume_id}_{smooth_kernel}_to_{sharp_kernel}.nii.gz')

    nib.save(nii_smooth, smooth_output_path)
    nib.save(nii_sharp,  sharp_output_path)

    print(f'Saved: {os.path.basename(smooth_output_path)}, {os.path.basename(sharp_output_path)}')


for idx in range(len(dataset)):
    print(f'\nProcessing volume {idx+1}/{len(dataset)}')
    sample = dataset[idx]
    reconstruct_volume(sample, model_s2h, model_h2s, device='cuda', output_dir=output_dir)

print(f'\nDone! All files saved to: {output_dir}')
