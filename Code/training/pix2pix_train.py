from models.pix2pix import Pix2PixModel
import torch
import os
import time
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Subset
import csv
from data.PSDDataset import PSDDataset

class Pix2PixOptions:
    def __init__(self):
        self.name = 'pix2pix'

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def train():
    opt = Pix2PixOptions()
    model = Pix2PixModel(opt)
    model.setup(opt)

    root_dir = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
    dataset = PSDDataset(root_dir = root_dir)
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(1000, len(dataset)))
    dataset = Subset(dataset, indices)
    img_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,num_workers=4, pin_memory=True)
    num_epochs = opt.num_epochs
    device = opt.device

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

        model.update_learning_rate()
    csv_file.close()
    print(f"\nTraining complete. Total time: {format_time(time.time() - train_start)}")


if __name__ == "__main__":
    train()
