from models.pix2pix import Pix2PixModel
import torch
import os
import time
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import csv
from data.PSDDataset import PSDDataset

import torch

class Pix2PixOptions:
    def __init__(self):
        self.name = 'pix2pix'
        self.isTrain = True
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

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def train():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    opt = Pix2PixOptions()
    opt.device = torch.device(f"cuda:{local_rank}") 
    opt.batch_size = 4
    model = Pix2PixModel(opt)
    model.setup(opt)

    root_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
    dataset = PSDDataset(root_dir = root_dir)
    
    # random.seed(42)
    # indices = random.sample(range(len(dataset)), min(1000, len(dataset)))
    # dataset = Subset(dataset, indices)
    sampler = DistributedSampler(dataset)
    img_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,num_workers=4, pin_memory=True, sampler = sampler)
    num_epochs = opt.num_epochs
    device = opt.device
    is_main_process = (local_rank == 0)

    if is_main_process:
        os.makedirs(opt.checkpoints_dir,exist_ok=True)
        csv_path = os.path.join(opt.checkpoints_dir, 'pix2pix_training_losses_vanilla.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        headers = ['Epoch', 'Batch'] + model.loss_names
        csv_writer.writerow(headers)

    train_start = time.time()
    epoch_iter = range(opt.epoch_count, opt.num_epochs + 1)
    if is_main_process:
        epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

    for epoch in epoch_iter:
        epoch_start = time.time()
        sampler.set_epoch(epoch)

        batch_iter = img_loader
        if is_main_process:
            batch_iter = tqdm(img_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", leave=False)

        for batch_idx, (I_smooth, I_sharp, _, _) in enumerate(batch_iter):
            I_smooth = I_smooth.to(device, non_blocking=True)
            I_sharp  = I_sharp.to(device, non_blocking=True)

            data = {
                'A': I_smooth,
                'B': I_sharp,
                'A_paths': '',
                'B_paths': ''
            }

            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            if is_main_process:
                batch_iter.set_postfix({k: f"{v:.3f}" for k, v in losses.items()})

            if is_main_process and batch_idx % 10 == 0:
                row = [epoch, batch_idx] + [losses[k] for k in model.loss_names]
                csv_writer.writerow(row)
                csv_file.flush()

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        eta = (num_epochs - epoch - 1) * epoch_time
        if is_main_process:
            epoch_iter.set_postfix({
                "epoch_time": format_time(epoch_time),
                "ETA": format_time(eta),
                "elapsed": format_time(elapsed)
            })

        if epoch % 10 == 0:
            model.save_networks(str(epoch))
            model.save_networks('latest')

        model.update_learning_rate()
    if is_main_process:
        csv_file.close()
    dist.destroy_process_group()
    print(f"\nTraining complete. Total time: {format_time(time.time() - train_start)}")


if __name__ == "__main__":
    train()
