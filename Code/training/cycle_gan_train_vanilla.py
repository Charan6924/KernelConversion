"""
Vanilla (single-GPU, no DDP) training loop for CycleGAN.
Use this for debugging or when DDP is not available.

Usage:
    uv run python Code/training/cycle_gan_train_vanilla.py
"""
import torch
import csv
import os
import time
import random
from torch.utils.data import DataLoader, Subset
from models.cycle_gan_model import CycleGANModel
from data.PSDDataset import PSDDataset
from tqdm import tqdm


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
        self.load_size = 286
        self.crop_size = 256
        self.max_dataset_size = float('inf')
        self.no_flip = False
        self.display_winsize = 256
        self.random_scale_max = 3.0
        self.epoch = 'latest'
        self.verbose = False
        self.suffix = ''
        self.isTrain = True
        self.phase = 'train'
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


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = CycleGANOptions()
    opt.gpu_ids = [0] if torch.cuda.is_available() else []
    num_epochs = opt.n_epochs + opt.n_epochs_decay

    model = CycleGANModel(opt)
    model.setup(opt)

    root_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
    dataset = PSDDataset(root_dir=root_dir)

    # Subsample for feasible training
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(500, len(dataset)))
    dataset = Subset(dataset, indices)

    img_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # CSV logging
    csv_path = os.path.join(opt.checkpoints_dir, 'cyclegan_training_losses_vanilla.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    headers = ['Epoch', 'Batch'] + model.loss_names
    csv_writer.writerow(headers)

    train_start = time.time()
    epoch_iter = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_iter:
        epoch_start = time.time()
        model.update_learning_rate()

        batch_iter = tqdm(img_loader, desc=f"Epoch {epoch}/{num_epochs}",
                          unit="batch", leave=False)

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

    csv_file.close()
    print(f"\nTraining complete. Total time: {format_time(time.time() - train_start)}")


if __name__ == "__main__":
    train()
