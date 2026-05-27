import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from data.PSDDataset import PSDDataset
from models.filterModel import FilterEstimator
from models.KernelEstimator import KernelEstimator
from utils.utils import compute_gradient_norm, load_checkpoint, compute_psd, compute_fft, generate_images, radial_to_2d
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainConfig:
    image_root:     str   = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
    lr:             float = 3e-5
    lambda_recon:   float = 0.1
    epochs:         int   = 100
    batch_size:     int   = 16
    resume:         bool  = False
    sched_factor:   float = 0.3
    sched_patience: int   = 3
    sched_min_lr:   float = 1e-7


def setup_ddp():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))
    return local_rank, dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


def all_reduce_stats(stats: dict) -> dict:
    for k, v in stats.items():
        t = torch.tensor(v, dtype=torch.float32, device='cuda')
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        stats[k] = t.item()
    return stats


def train_one_epoch(model, ft_model, image_loader, optimizer, scaler, lambda_recon, device, epoch, is_main):
    model.train()
    ft_model.eval()

    running_loss  = 0.0
    running_ft    = 0.0
    running_recon = 0.0
    running_grad  = 0.0
    n_batches     = 0

    l1_loss = nn.L1Loss()
    loader  = tqdm(image_loader, desc="Training", unit="batch") if is_main else image_loader

    for I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2 in loader:
        I_smooth_1 = I_smooth_1.to(device, non_blocking=True)
        I_sharp_1  = I_sharp_1.to(device, non_blocking=True)
        I_smooth_2 = I_smooth_2.to(device, non_blocking=True)
        I_sharp_2  = I_sharp_2.to(device, non_blocking=True)

        with torch.no_grad():
            psd_smooth = compute_psd(I_smooth_1, device='cuda').to(device, non_blocking=True)
            psd_sharp  = compute_psd(I_sharp_2,  device='cuda').to(device, non_blocking=True)

            teacher_s2sh, teacher_sh2s = ft_model(psd_smooth, psd_sharp)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            out_smooth = model(psd_smooth)
            out_sharp  = model(psd_sharp)

            k_smooth = radial_to_2d(out_smooth, grid_size=512)
            k_sharp  = radial_to_2d(out_sharp,  grid_size=512)
            filt_s2sh = k_sharp / (k_smooth + 1e-10)
            filt_sh2s = k_smooth / (k_sharp + 1e-10)

            ft_loss = l1_loss(filt_s2sh, teacher_s2sh) + l1_loss(filt_sh2s, teacher_sh2s)

            I_gen_sharp, I_gen_smooth = generate_images(
                I_smooth_1, I_sharp_2, filt_s2sh, filt_sh2s, device
            )
            recon_loss = (
                F.l1_loss(I_gen_sharp,  I_sharp_1.squeeze(1).float()) +
                F.l1_loss(I_gen_smooth, I_smooth_2.squeeze(1).float())
            )

            loss = ft_loss + lambda_recon * recon_loss

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            grad_norm = compute_gradient_norm(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            grad_norm = compute_gradient_norm(model)
            optimizer.step()

        running_loss  += loss.item()
        running_ft    += ft_loss.item()
        running_recon += recon_loss.item()
        running_grad  += grad_norm
        n_batches     += 1

    denom = max(n_batches, 1)
    stats = {
        'total_loss': running_loss  / denom,
        'ft_loss':    running_ft    / denom,
        'recon_loss': running_recon / denom,
        'grad_norm':  running_grad  / denom,
    }
    plot_data = {
        'I_gen_sharp':  I_gen_sharp.detach().cpu(),
        'I_gen_smooth': I_gen_smooth.detach().cpu(),
        'I_sharp_1':    I_sharp_1.detach().cpu(),
        'I_smooth_2':   I_smooth_2.detach().cpu(),
        'filt_s2sh':    filt_s2sh.detach().cpu(),
        'filt_sh2s':    filt_sh2s.detach().cpu(),
        'real_s2sh':    teacher_s2sh.detach().cpu(),
        'real_sh2s':    teacher_sh2s.detach().cpu(),
    }
    return stats, plot_data


@torch.no_grad()
def validate(model, ft_model, image_loader, lambda_recon, device):
    model.eval()

    total_loss  = 0.0
    total_ft    = 0.0
    total_recon = 0.0
    num_batches = 0

    l1_loss = nn.L1Loss()

    for I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2 in image_loader:
        I_smooth_1 = I_smooth_1.to(device, non_blocking=True)
        I_sharp_1  = I_sharp_1.to(device, non_blocking=True)
        I_smooth_2 = I_smooth_2.to(device, non_blocking=True)
        I_sharp_2  = I_sharp_2.to(device, non_blocking=True)

        psd_smooth = compute_psd(I_smooth_1, device='cuda').to(device, non_blocking=True)
        psd_sharp  = compute_psd(I_sharp_2,  device='cuda').to(device, non_blocking=True)

        teacher_s2sh, teacher_sh2s = ft_model(psd_smooth, psd_sharp)

        out_smooth = model(psd_smooth)
        out_sharp  = model(psd_sharp)

        k_smooth = radial_to_2d(out_smooth, grid_size=512)
        k_sharp  = radial_to_2d(out_sharp,  grid_size=512)

        filt_s2sh = k_sharp / (k_smooth + 1e-10)
        filt_sh2s = k_smooth / (k_sharp + 1e-10)

        ft_loss = l1_loss(filt_s2sh, teacher_s2sh) + l1_loss(filt_sh2s, teacher_sh2s)

        I_gen_sharp, I_gen_smooth = generate_images(
            I_smooth_1, I_sharp_2, filt_s2sh, filt_sh2s, device
        )
        recon_loss = (
            F.l1_loss(I_gen_sharp,  I_sharp_1.squeeze(1).float()) +
            F.l1_loss(I_gen_smooth, I_smooth_2.squeeze(1).float())
        )

        total_loss  += (ft_loss + lambda_recon * recon_loss).item()
        total_ft    += ft_loss.item()
        total_recon += recon_loss.item()
        num_batches += 1

    denom = max(num_batches, 1)
    return {
        'total_loss': total_loss  / denom,
        'ft_loss':    total_ft    / denom,
        'recon_loss': total_recon / denom,
    }


def plot_epoch_results(plot_data, epoch, out_dir):
    plot_data = {k: v.float() if isinstance(v, torch.Tensor) else v
                 for k, v in plot_data.items()}

    vis_dir = out_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'Epoch {epoch} — KernelEstimator vs Teacher Filters', fontsize=14)

    f_s2sh = plot_data['filt_s2sh'][0].numpy()
    f_sh2s = plot_data['filt_sh2s'][0].numpy()
    r_s2sh = plot_data['real_s2sh'][0].numpy()
    r_sh2s = plot_data['real_sh2s'][0].numpy()
    mid    = f_s2sh.shape[0] // 2

    im0 = axes[0, 0].imshow(f_s2sh, cmap='hot')
    axes[0, 0].set_title('Pred filter  smooth→sharp')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(f_sh2s, cmap='hot')
    axes[0, 1].set_title('Pred filter  sharp→smooth')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].plot(f_s2sh[mid], label='pred s->sh', color='tomato',    linewidth=1.5)
    axes[0, 2].plot(f_sh2s[mid], label='pred sh->s', color='steelblue', linewidth=1.5)
    axes[0, 2].set_title('Predicted - central row profile')
    axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    im3 = axes[1, 0].imshow(r_s2sh, cmap='hot')
    axes[1, 0].set_title('Teacher filter  smooth->sharp')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(r_sh2s, cmap='hot')
    axes[1, 1].set_title('Teacher filter  sharp->smooth')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].plot(r_s2sh[mid], label='teacher s->sh', color='tomato',    linewidth=1.5, linestyle='--')
    axes[1, 2].plot(r_sh2s[mid], label='teacher sh->s', color='steelblue', linewidth=1.5, linestyle='--')
    axes[1, 2].set_title('Teacher - central row profile')
    axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

    I_gen_sharp  = plot_data['I_gen_sharp'][0].squeeze().numpy()
    I_gen_smooth = plot_data['I_gen_smooth'][0].squeeze().numpy()
    I_sharp_1    = plot_data['I_sharp_1'][0].squeeze().numpy()
    I_smooth_2   = plot_data['I_smooth_2'][0].squeeze().numpy()

    axes[2, 0].imshow(I_gen_sharp,  cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('Generated sharp'); axes[2, 0].axis('off')

    axes[2, 1].imshow(I_gen_smooth, cmap='gray', vmin=0, vmax=1)
    axes[2, 1].set_title('Generated smooth'); axes[2, 1].axis('off')

    diff_sharp  = np.abs(I_gen_sharp  - I_sharp_1)
    diff_smooth = np.abs(I_gen_smooth - I_smooth_2)
    axes[2, 2].plot(diff_sharp[mid],  label='|sharp diff|',  color='orange')
    axes[2, 2].plot(diff_smooth[mid], label='|smooth diff|', color='green')
    axes[2, 2].set_title('Central row |recon error|')
    axes[2, 2].legend(); axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(vis_dir / f'epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    local_rank, world_size = setup_ddp()
    device  = f'cuda:{local_rank}'
    is_main = (local_rank == 0)

    cfg = TrainConfig()

    out_dir  = Path("training_output_kernel2d")
    ckpt_dir = out_dir / "checkpoints"

    if is_main:
        for d in [out_dir, ckpt_dir]:
            d.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    if is_main:
        csv_path = out_dir / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'learning_rate',
            'train_total_loss', 'train_ft_loss', 'train_recon_loss', 'train_grad_norm',
            'val_total_loss', 'val_ft_loss', 'val_recon_loss',
        ])

    if is_main:
        print(f"Device: {device}  |  world_size: {world_size}  |  lr={cfg.lr}  |  epochs={cfg.epochs}")
        print(f"Model: KernelEstimator (256 radial profile -> radial_to_2d -> ratio), distilled from FilterEstimator")
        print(cfg)

    img_dataset = PSDDataset(root_dir=cfg.image_root, preload=False)
    n_train = int(0.9 * len(img_dataset))
    img_train, img_val = random_split(
        img_dataset, [n_train, len(img_dataset) - n_train],
        generator=torch.Generator().manual_seed(42)
    )

    img_train_sampler = DistributedSampler(img_train, num_replicas=world_size, rank=local_rank, shuffle=True)
    img_val_sampler   = DistributedSampler(img_val,   num_replicas=world_size, rank=local_rank, shuffle=False)

    img_train_loader = DataLoader(img_train, batch_size=cfg.batch_size, sampler=img_train_sampler, num_workers=0, pin_memory=False)
    img_val_loader   = DataLoader(img_val,   batch_size=cfg.batch_size, sampler=img_val_sampler,   num_workers=0, pin_memory=False)

    if is_main:
        print(f"Images - train: {len(img_train)}, val: {len(img_val)}")

    ft_model = FilterEstimator().to(device)
    ft_checkpoint = torch.load(
        '/home/cxv166/KernelConversionResearch/training_filter_model/checkpoints/epoch_17.pth',
        map_location=device
    )
    ft_model.load_state_dict(ft_checkpoint['model_state_dict'])
    ft_model.eval()
    for p in ft_model.parameters():
        p.requires_grad_(False)

    model = KernelEstimator().to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.sched_factor,
        patience=cfg.sched_patience, min_lr=cfg.sched_min_lr
    )
    scaler = torch.amp.GradScaler('cuda')

    if is_main:
        print(f"KernelEstimator params: {sum(p.numel() for p in model.parameters()):,}")

    start_epoch = 0
    best_val    = float('inf')

    if cfg.resume:
        ckpt_path = ckpt_dir / "latest_checkpoint.pth"
        loaded = load_checkpoint(ckpt_path, model.module, optimizer, scaler)
        if loaded:
            start_epoch = loaded['epoch'] + 1
            best_val    = loaded['best_val_loss']
            if 'scheduler_state_dict' in loaded:
                scheduler.load_state_dict(loaded['scheduler_state_dict'])

    for epoch in range(start_epoch, cfg.epochs):
        ep = epoch + 1
        img_train_sampler.set_epoch(epoch)
        cur_lr = optimizer.param_groups[0]['lr']

        if is_main:
            print(f"\n--- Epoch {ep}/{cfg.epochs}  (lr={cur_lr:.2e}) ---")

        train_stats, plot_data = train_one_epoch(
            model, ft_model, img_train_loader,
            optimizer, scaler, cfg.lambda_recon,
            device, epoch=ep, is_main=is_main
        )
        val_stats = validate(model, ft_model, img_val_loader, cfg.lambda_recon, device)

        train_stats = all_reduce_stats(train_stats)
        val_stats   = all_reduce_stats(val_stats)

        if is_main:
            plot_epoch_results(plot_data, ep, out_dir)
            scheduler.step(val_stats['total_loss'])
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr < cur_lr:
                print(f"LR dropped: {cur_lr:.2e} -> {new_lr:.2e}")

            csv_writer.writerow([
                ep, new_lr,
                train_stats['total_loss'], train_stats['ft_loss'], train_stats['recon_loss'], train_stats['grad_norm'],
                val_stats['total_loss'],   val_stats['ft_loss'],   val_stats['recon_loss'],
            ])
            csv_file.flush()

            print(
                f"  train - total: {train_stats['total_loss']:.4f}  ft: {train_stats['ft_loss']:.4f}"
                f"  recon: {train_stats['recon_loss']:.4f}  grad: {train_stats['grad_norm']:.4f}"
            )
            print(
                f"  val   - total: {val_stats['total_loss']:.4f}  ft: {val_stats['ft_loss']:.4f}"
                f"  recon: {val_stats['recon_loss']:.4f}"
            )

            is_best = val_stats['total_loss'] < best_val
            if is_best:
                best_val = val_stats['total_loss']
                print(f"  ** new best val loss: {best_val:.6f} **")

            ckpt = {
                'epoch':                ep,
                'model_state_dict':     model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict':    scaler.state_dict(),
                'best_val_loss':        best_val,
                'lambda_recon':         cfg.lambda_recon,
                'learning_rate':        cfg.lr,
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{ep}_checkpoint.pth")
            if is_best:
                torch.save(ckpt, ckpt_dir / "best_checkpoint.pth")
        else:
            scheduler.step(val_stats['total_loss'])

        dist.barrier()

    if is_main:
        csv_file.close()
        print(f"\nDone. Best val loss: {best_val:.6f}")
        print(f"Metrics saved to: {csv_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
