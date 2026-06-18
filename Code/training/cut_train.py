import torch
import os
import csv
from tqdm import tqdm
from data.PSDDataset import PSDDataset
from models.cut_model import CUTModel
from data.PSDDataset import PSDDataset
from torch.utils.data import DataLoader, Subset
import time
import random

class CUTTrainOptions:
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
        self.batch_size = 2
        self.max_dataset_size = float('inf')
        self.no_flip = False
        self.display_winsize = 256
        self.random_scale_max = 3.0
        
        self.epoch = 'latest'
        self.verbose = False
        self.suffix = ''
        self.isTrain = True
        self.phase = 'train'
        
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
        
        self.n_epochs = 50
        self.n_epochs_decay = 50  # 100 epochs total instead of 200
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = 0.0002
        self.gan_mode = 'lsgan'
        self.pool_size = 0  
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.load_size = 512
        self.crop_size = 512

        self.CUT_mode = 'CUT'
        self.lambda_GAN = 1.0
        self.lambda_smooth = 0.0
        self.lambda_NCE = 1.0  # Default for standard CUT
        self.nce_idt = True    # Default for standard CUT
        self.nce_layers = '0,4,8,12,16'
        self.nce_includes_all_negatives_from_minibatch = False
        self.netF = 'mlp_sample'
        self.netF_nc = 256
        self.nce_T = 0.07
        self.num_patches = 256
        self.flip_equivariance = False 
        self.device = 'cuda:0'
        self.lambda_spa_unsup_A = 1.0   # Weight for unsupervised spatial loss using optical flow
        self.lambda_kl = 1.0            # Weight for KL divergence structural consistency
        self.unsup_idt_spa = True       # Apply unsupervised spatial loss to identity mapping as well

        # Hyperparameters for Optical Flow Generation & Noise
        self.motion_level = 8.0         # Blur/scale factor for random optical flow generation
        self.shift_level = 10.0         # Spatial translation/pixel shift bounds
        self.scale_level = 0.0          # Zooming/scaling bounds for generated flow
        self.noise_level = 0.001


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def train():
    opt = CUTTrainOptions()
    model = CUTModel(opt=opt)
    model.setup(opt)
    device = 'cuda'
    num_epochs = opt.n_epochs

    root_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
    dataset = PSDDataset(root_dir = root_dir)
    # random.seed(42)
    # indices = random.sample(range(len(dataset)), min(1000, len(dataset)))
    # dataset = Subset(dataset, indices)
    img_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,num_workers=4, pin_memory=True)

    first_batch = next(iter(img_loader))
    dummy_data = {
        'A': first_batch[0].to(device),
        'B': first_batch[1].to(device),
        'A_paths': '',
        'B_paths': ''
    }
    model.data_dependent_initialize(dummy_data)

    csv_path = os.path.join(opt.checkpoints_dir, 'cyclegan_training_losses_vanilla.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    headers = ['Epoch', 'Batch'] + model.loss_names
    csv_writer.writerow(headers)

    train_start = time.time()
    epoch_iter = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_iter:
        epoch_start = time.time()
        

        batch_iter = tqdm(img_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", leave=False)

        for batch_idx, (I_smooth, I_sharp, _, _) in enumerate(batch_iter):
            I_smooth = I_smooth.to(device, non_blocking=True)
            I_sharp  = I_sharp.to(device, non_blocking=True)

            data = {
                'A': I_sharp,
                'B': I_smooth,
                'A_paths': '',
                'B_paths': ''
            }

            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            batch_iter.set_postfix({k: f"{v:.3f}" for k, v in losses.items()})

            if batch_idx % 10 == 0:
                row = [epoch, batch_idx] + [losses[k] for k in model.loss_names]
                csv_writer.writerow(row)
                csv_file.flush()

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        eta = (num_epochs - epoch - 1) * epoch_time
        epoch_iter.set_postfix({
            "epoch_time": format_time(epoch_time),
            "ETA": format_time(eta),
            "elapsed": format_time(elapsed)
        })

        if epoch % 10 == 0:
            model.save_networks(str(epoch))
            model.save_networks('latest')

        model.update_learning_rate()
    csv_file.close()
    print(f"\nTraining complete. Total time: {format_time(time.time() - train_start)}")


if __name__ == "__main__":
    train()
