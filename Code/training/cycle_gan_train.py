import torch
import csv
from torch.utils.data import DataLoader
from models.cycle_gan_model import CycleGANModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from data.PSDDataset import PSDDataset
import os
import time
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
        self.batch_size = 1          # 1 per GPU, 4 effective with 4 GPUs
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


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train():
    local_rank = setup_ddp()
    num_epochs = 50                  # reduced from 200
    device = torch.device(f"cuda:{local_rank}")

    opt = CycleGANOptions()
    opt.gpu_ids = [local_rank]

    model = CycleGANModel(opt)
    model.setup(opt)

    # Wrap each sub-network with DDP
    # find_unused_parameters handles inplace op issues
    # static_graph=True is safe for CycleGAN and speeds up communication
    for name in model.model_names:
        if isinstance(name, str):
            net = getattr(model, 'net' + name)
            net = net.to(device)
            net = DDP(net, device_ids=[local_rank],
                      static_graph=True)   
            setattr(model, 'net' + name, net)

    root_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
    dataset = PSDDataset(root_dir=root_dir)

    # Subsample to keep training feasible
    from torch.utils.data import Subset
    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(500, len(dataset)))
    dataset = Subset(dataset, indices)

    sampler = DistributedSampler(dataset, shuffle=True)
    img_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, sampler=sampler)

    # CSV logging only on rank 0
    if local_rank == 0:
        csv_path = os.path.join(opt.checkpoints_dir, 'cyclegan_training_losses.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        headers = ['Epoch', 'Batch'] + model.loss_names
        csv_writer.writerow(headers)

    train_start = time.time()

    # Epoch bar only on rank 0
    epoch_iter = tqdm(range(num_epochs), desc="Training", unit="epoch") if local_rank == 0 else range(num_epochs)

    for epoch in epoch_iter:
        sampler.set_epoch(epoch)    
        epoch_start = time.time()
        model.update_learning_rate()

        batch_iter = tqdm(img_loader, desc=f"Epoch {epoch}/{num_epochs}",
                          unit="batch", leave=False) if local_rank == 0 else img_loader

        for batch_idx, (I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2) in enumerate(batch_iter):
            I_smooth_1 = I_smooth_1.to(device, non_blocking=True)
            I_sharp_1  = I_sharp_1.to(device, non_blocking=True)

            data = {
                'A': I_smooth_1,
                'B': I_sharp_1,
                'A_paths': '',
                'B_paths': ''
            }

            model.set_input(data)
            model.optimize_parameters()

            if local_rank == 0:
                losses = model.get_current_losses()
                batch_iter.set_postfix({k: f"{v:.3f}" for k, v in losses.items()})

                if batch_idx % 10 == 0:
                    row = [epoch, batch_idx] + [losses[k] for k in model.loss_names]
                    csv_writer.writerow(row)
                    csv_file.flush()

        if local_rank == 0:
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

    if local_rank == 0:
        csv_file.close()
        print(f"\nTraining complete. Total time: {format_time(time.time() - train_start)}")

    cleanup_ddp()


if __name__ == "__main__":
    train()
