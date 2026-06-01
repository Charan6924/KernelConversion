import torch
import csv
from torch.utils.data import DataLoader 
from models.cycle_gan_model import CycleGANModel
from data.PSDDataset import PSDDataset
import os

class CycleGANOptions:
    def __init__(self):
        # ----------------------------------------
        # Base Options (Merged & Protected)
        # ----------------------------------------
        self.dataroot = 'placeholder'
        self.name = 'Cycle-GAN'
        self.easy_label = 'Cycle-GAN'
        self.gpu_ids = [0]
        self.checkpoints_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_cyclegan'
        self.model = 'cycle_gan'
        
        # CRITICAL PROTECTED OVERRIDES FOR NIFTI
        self.input_nc = 1           # Protected: Must be 1 for Grayscale
        self.output_nc = 1          # Protected: Must be 1 for Grayscale
        self.preprocess = 'none'    # Protected: Bypass PIL resizing
        
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
        
        # Dataset attributes (Mostly bypassed by your custom DataLoader)
        self.dataset_mode = 'unaligned'
        self.direction = 'AtoB'
        self.serial_batches = False
        self.num_threads = 4
        self.batch_size = 16        # Matched to your DataLoader batch_size
        self.load_size = 286
        self.crop_size = 256
        self.max_dataset_size = float('inf')
        self.no_flip = False
        self.display_winsize = 256
        self.random_scale_max = 3.0
        
        self.epoch = 'latest'
        self.verbose = False
        self.suffix = ''

        # ----------------------------------------
        # Train Options (Merged)
        # ----------------------------------------
        self.isTrain = True
        self.phase = 'train'
        self.amp = False            # Protected: Disabled for standard PyTorch
        
        # Visdom and HTML visualization parameters
        self.display_freq = 400
        self.display_ncols = 4
        self.display_id = None
        self.display_server = 'http://localhost'
        self.display_env = 'main'
        self.display_port = 8097
        self.update_html_freq = 1000
        self.print_freq = 100
        self.no_html = False
        
        # Network saving and loading parameters
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

def train():
    num_epochs = 200
    device = 'cuda'
    opt = CycleGANOptions()
    
    model = CycleGANModel(opt)
    model.setup(opt) 
    
    root_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
    dataset = PSDDataset(root_dir = root_dir)
    img_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    csv_path = os.path.join('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_cyclegan','cyclegan_training_losses.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    headers = ['Epoch', 'Batch'] + model.loss_names
    csv_writer.writerow(headers)

    for epoch in range(num_epochs):
        model.update_learning_rate()
        
        for batch_idx, (I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2) in enumerate(img_loader):
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

            losses = model.get_current_losses()
                
            if batch_idx % 50 == 0:
                row = [epoch, batch_idx] + [losses[k] for k in model.loss_names]
                csv_writer.writerow(row)
                csv_file.flush() 
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(img_loader)}] || {loss_str}")

        if epoch % 10 == 0:
            model.save_networks(str(epoch))
            model.save_networks('latest')
            
    csv_file.close()

if __name__ == "__main__":
    train()
