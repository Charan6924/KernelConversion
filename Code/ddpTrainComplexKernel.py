import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from PSDDataset import PSDDataset
from KernelEstimator import KernelEstimator
from utils import (
    compute_gradient_norm, load_checkpoint,
    compute_psd, compute_fft, huber
)
from Discriminator import MultiScaleDiscriminator, lsgan_d_loss, lsgan_g_loss  # type: ignore
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainConfig:
    image_root:    str   = r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root"
    alpha:         float = 0.5
    lr:            float = 1e-4
    d_lr:          float = 2e-4
    lambda_adv:    float = 0.1
    epochs:        int   = 100
    batch_size:    int   = 16
    resume:        bool  = False
    sched_factor:  float = 0.5
    sched_patience: int  = 5
    sched_min_lr:  float = 1e-7

def setup_ddp():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))
    return local_rank, dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


def all_reduce_stats(stats: dict) -> dict:
    for k, v in stats.items():
        if k == 'nan_batches':
            continue
        t = torch.tensor(v, dtype=torch.float32, device='cuda')
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        stats[k] = t.item()
    return stats

def model_to_complex_kernel(out: torch.Tensor) -> torch.Tensor:
    out = out.float()
    return torch.complex(out[:, 0], out[:, 1])


def apply_complex_filter(image: torch.Tensor,kernel: torch.Tensor,device: str) -> torch.Tensor:
    if image.dim() == 4:
        image = image.squeeze(1)                          

    I_fft = torch.fft.fftshift(torch.fft.fft2(image.float()))
    filtered = torch.fft.ifft2(torch.fft.ifftshift(I_fft * kernel))          
    return filtered.real                                 


def complex_kernel_ratio(H_num: torch.Tensor,H_den: torch.Tensor,eps: float = 1e-10) -> torch.Tensor:
    denom = H_den + eps * torch.exp(1j * torch.angle(H_den))
    return H_num / denom


def train_one_epoch(model, D_sharp, D_smooth, image_loader,optimizer, opt_D, scaler, alpha, lambda_adv,device, epoch, is_main):
    model.train()
    D_sharp.train()
    D_smooth.train()

    running_loss   = 0.0
    running_d_loss = 0.0
    running_g_adv  = 0.0
    running_ft     = 0.0
    running_grad   = 0.0
    n_batches      = 0
    skipped        = 0

    loader = tqdm(image_loader, desc="Training", unit="batch") if is_main else image_loader

    for i, (I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2) in enumerate(loader):
        I_smooth_1 = I_smooth_1.to(device, non_blocking=True)
        I_sharp_1  = I_sharp_1.to(device, non_blocking=True)
        I_smooth_2 = I_smooth_2.to(device, non_blocking=True)
        I_sharp_2  = I_sharp_2.to(device, non_blocking=True)

        with torch.no_grad():
            psd_smooth   = compute_psd(I_smooth_1, device='cuda').to(device, non_blocking=True)
            psd_sharp    = compute_psd(I_sharp_2,  device='cuda').to(device, non_blocking=True)
            I_smooth_fft = compute_fft(I_smooth_1)   # (B, H, W) complex
            I_sharp_fft  = compute_fft(I_sharp_1)    # (B, H, W) complex

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):  # type: ignore
            out_smooth = model(psd_smooth)   # (B, 2, 512, 512)
            out_sharp  = model(psd_sharp)    # (B, 2, 512, 512)

            H_smooth = model_to_complex_kernel(out_smooth)  # (B, 512, 512) complex
            H_sharp  = model_to_complex_kernel(out_sharp)   # (B, 512, 512) complex

            filt_s2sh = complex_kernel_ratio(H_sharp,  H_smooth)  # smooth → sharp
            filt_sh2s = complex_kernel_ratio(H_smooth, H_sharp)   # sharp  → smooth

            I_gen_sharp  = apply_complex_filter(I_smooth_1, filt_s2sh, device)  # (B, H, W)
            I_gen_smooth = apply_complex_filter(I_sharp_2,  filt_sh2s, device)  # (B, H, W)

        I_gen_sharp_4d  = I_gen_sharp.unsqueeze(1)    # (B, 1, H, W) for discriminator
        I_gen_smooth_4d = I_gen_smooth.unsqueeze(1)

        opt_D.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):  # type: ignore
            pred_real_sharp  = D_sharp(I_sharp_1)
            pred_fake_sharp  = D_sharp(I_gen_sharp_4d.detach())
            pred_real_smooth = D_smooth(I_smooth_2)
            pred_fake_smooth = D_smooth(I_gen_smooth_4d.detach())

            d_loss = (lsgan_d_loss(pred_real_sharp,  pred_fake_sharp) +
                      lsgan_d_loss(pred_real_smooth, pred_fake_smooth))

        if scaler:
            scaler.scale(d_loss).backward()
            scaler.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(
                list(D_sharp.parameters()) + list(D_smooth.parameters()), max_norm=1.0
            )
            scaler.step(opt_D)
        else:
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(D_sharp.parameters()) + list(D_smooth.parameters()), max_norm=1.0
            )
            opt_D.step()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):  # type: ignore
            pred_fake_sharp  = D_sharp(I_gen_sharp_4d)
            pred_fake_smooth = D_smooth(I_gen_smooth_4d)
            g_adv_loss = (lsgan_g_loss(pred_fake_sharp) +
                          lsgan_g_loss(pred_fake_smooth))

            # Fourier loss: log|H_sharp| - log|H_smooth| should match
            # log|FFT(I_sharp)| - log|FFT(I_smooth)|
            ft_loss = huber(
                torch.log(I_sharp_fft.abs() + 1e-7) - torch.log(I_smooth_fft.abs() + 1e-7),
                torch.log(H_sharp.abs() + 1e-7) - torch.log(H_smooth.abs() + 1e-7)
            )

            loss = ft_loss + lambda_adv * g_adv_loss

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm = compute_gradient_norm(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm = compute_gradient_norm(model)
            optimizer.step()

        running_loss += loss.item()
        running_d_loss += d_loss.item()
        running_g_adv += g_adv_loss.item()
        running_ft += ft_loss.item()
        running_grad += grad_norm
        n_batches += 1

    if skipped > 0 and is_main:
        print(f"WARNING: {skipped} batches were skipped (NaN/Inf)")

    denom = max(n_batches, 1)
    stats = {
        'total_loss':  running_loss   / denom,
        'd_loss':      running_d_loss / denom,
        'g_adv_loss':  running_g_adv  / denom,
        'ft_loss':     running_ft     / denom,
        'grad_norm':   running_grad   / denom,
        'nan_batches': skipped,
    }

    plot_data = {
        'I_gen_sharp':  I_gen_sharp.detach().cpu(),   # type: ignore
        'I_gen_smooth': I_gen_smooth.detach().cpu(),  # type: ignore
        'I_sharp_1':    I_sharp_1.detach().cpu(),     # type: ignore
        'I_smooth_2':   I_smooth_2.detach().cpu(),    # type: ignore
        'H_smooth':     H_smooth.detach().cpu(),      # type: ignore
        'H_sharp':      H_sharp.detach().cpu(),       # type: ignore
        'filt_s2sh':    filt_s2sh.detach().cpu(),     # type: ignore
        'filt_sh2s':    filt_sh2s.detach().cpu(),     # type: ignore
    }
    return stats, plot_data



@torch.no_grad()
def validate_adv(model, D_sharp, D_smooth, image_loader,
                 alpha, lambda_adv, device):
    model.eval()
    D_sharp.eval()
    D_smooth.eval()

    total_loss   = 0.0
    total_g_adv  = 0.0
    total_ft     = 0.0
    num_batches  = 0

    for I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2 in image_loader:
        I_smooth_1 = I_smooth_1.to(device, non_blocking=True)
        I_sharp_1  = I_sharp_1.to(device, non_blocking=True)
        I_smooth_2 = I_smooth_2.to(device, non_blocking=True)
        I_sharp_2  = I_sharp_2.to(device, non_blocking=True)

        psd_smooth   = compute_psd(I_smooth_1, device='cuda').to(device, non_blocking=True)
        psd_sharp    = compute_psd(I_sharp_2,  device='cuda').to(device, non_blocking=True)
        I_smooth_fft = compute_fft(I_smooth_1)
        I_sharp_fft  = compute_fft(I_sharp_1)

        out_smooth = model(psd_smooth)
        out_sharp  = model(psd_sharp)

        H_smooth = model_to_complex_kernel(out_smooth)
        H_sharp  = model_to_complex_kernel(out_sharp)

        filt_s2sh = complex_kernel_ratio(H_sharp,  H_smooth)
        filt_sh2s = complex_kernel_ratio(H_smooth, H_sharp)

        I_gen_sharp  = apply_complex_filter(I_smooth_1, filt_s2sh, device)
        I_gen_smooth = apply_complex_filter(I_sharp_2,  filt_sh2s, device)

        I_gen_sharp_4d  = I_gen_sharp.unsqueeze(1)
        I_gen_smooth_4d = I_gen_smooth.unsqueeze(1)

        g_adv_loss = (lsgan_g_loss(D_sharp(I_gen_sharp_4d)) +
                      lsgan_g_loss(D_smooth(I_gen_smooth_4d)))

        ft_loss = huber(
            torch.log(I_sharp_fft.abs()  + 1e-7) - torch.log(I_smooth_fft.abs() + 1e-7),
            torch.log(H_sharp.abs()      + 1e-7) - torch.log(H_smooth.abs()     + 1e-7)
        )

        batch_loss = ft_loss + lambda_adv * g_adv_loss

        total_loss  += batch_loss.item()
        total_g_adv += g_adv_loss.item()
        total_ft    += ft_loss.item()
        num_batches += 1

    denom = max(num_batches, 1)
    return {
        'total_loss': total_loss  / denom,
        'g_adv_loss': total_g_adv / denom,
        'ft_loss':    total_ft    / denom,
    }



def plot_epoch_results(plot_data, epoch, out_dir):
    plot_data = {k: v.float() if isinstance(v, torch.Tensor) else v
                 for k, v in plot_data.items()}

    vis_dir = out_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'Epoch {epoch}', fontsize=14)

    # Magnitude of estimated kernels (log scale, central row)
    H_smooth_mag = plot_data['H_smooth'][0].abs().numpy()
    H_sharp_mag  = plot_data['H_sharp'][0].abs().numpy()
    mid = H_smooth_mag.shape[0] // 2
    axes[0, 0].plot(np.log(H_smooth_mag[mid] + 1e-7), label='Smooth |H|', color='blue')
    axes[0, 0].plot(np.log(H_sharp_mag[mid]  + 1e-7), label='Sharp  |H|', color='red')
    axes[0, 0].set_title('Log-magnitude of estimated kernels (central row)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Phase of estimated kernels (central row)
    axes[0, 1].plot(torch.angle(plot_data['H_smooth'][0]).numpy()[mid], label='Smooth phase', color='blue')
    axes[0, 1].plot(torch.angle(plot_data['H_sharp'][0]).numpy()[mid],  label='Sharp  phase', color='red')
    axes[0, 1].set_title('Phase of estimated kernels (central row)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Magnitude of conversion filters (central row)
    filt_s2sh_mag = plot_data['filt_s2sh'][0].abs().numpy()
    axes[1, 0].plot(np.log(filt_s2sh_mag[mid] + 1e-7), color='green')
    axes[1, 0].set_title('Log-magnitude: filter smooth→sharp (central row)')
    axes[1, 0].grid(True, alpha=0.3)

    filt_sh2s_mag = plot_data['filt_sh2s'][0].abs().numpy()
    axes[1, 1].plot(np.log(filt_sh2s_mag[mid] + 1e-7), color='purple')
    axes[1, 1].set_title('Log-magnitude: filter sharp→smooth (central row)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(vis_dir / f'epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    local_rank, world_size = setup_ddp()
    device  = f'cuda:{local_rank}'
    is_main = (local_rank == 0)

    cfg = TrainConfig()

    out_dir  = Path(f"training_output_{cfg.alpha}")
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
            'train_total_loss', 'train_d_loss', 'train_g_adv_loss',
            'train_ft_loss', 'train_grad_norm', 'nan_batches',
            'val_total_loss', 'val_g_adv_loss', 'val_ft_loss',
        ])

    if is_main:
        print(f"Device: {device}  |  world_size: {world_size}  |  alpha={cfg.alpha}  |  lr={cfg.lr}  |  epochs={cfg.epochs}")
        print(cfg)

    img_dataset = PSDDataset(root_dir=cfg.image_root, preload=False)
    n_train = int(0.9 * len(img_dataset))
    img_train, img_val = random_split(
        img_dataset, [n_train, len(img_dataset) - n_train],
        generator=torch.Generator().manual_seed(42)
    )

    img_train_sampler = DistributedSampler(img_train, num_replicas=world_size, rank=local_rank, shuffle=True)
    img_val_sampler = DistributedSampler(img_val,   num_replicas=world_size, rank=local_rank, shuffle=False)

    img_train_loader = DataLoader(img_train, batch_size=cfg.batch_size, sampler=img_train_sampler, num_workers=0, pin_memory=False)
    img_val_loader = DataLoader(img_val,   batch_size=cfg.batch_size, sampler=img_val_sampler,   num_workers=0, pin_memory=False)

    if is_main:
        print(f"Images — train: {len(img_train)}, val: {len(img_val)}")

    model = KernelEstimator().to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    D_sharp = MultiScaleDiscriminator(input_nc=1, ndf=64, n_layers=3, num_scales=3).to(device)
    D_smooth = MultiScaleDiscriminator(input_nc=1, ndf=64, n_layers=3, num_scales=3).to(device)
    D_sharp = DDP(D_sharp,  device_ids=[local_rank], broadcast_buffers=False)
    D_smooth = DDP(D_smooth, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    opt_D = torch.optim.Adam(
        list(D_sharp.parameters()) + list(D_smooth.parameters()),
        lr=cfg.d_lr, betas=(0.5, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.sched_factor,
        patience=cfg.sched_patience, min_lr=cfg.sched_min_lr
    )
    scaler = torch.amp.GradScaler('cuda')  # type: ignore

    if is_main:
        d_params = (sum(p.numel() for p in D_sharp.parameters()) +
                    sum(p.numel() for p in D_smooth.parameters()))
        print(f"Generator params:      {sum(p.numel() for p in model.parameters()):,}")
        print(f"Discriminator params:  {d_params:,}")

    start_epoch = 0
    best_val    = float('inf')

    if cfg.resume:
        ckpt_path = ckpt_dir / "latest_checkpoint.pth"
        raw_model = model.module
        loaded = load_checkpoint(ckpt_path, raw_model, optimizer, scaler)
        if loaded:
            start_epoch = loaded['epoch'] + 1
            best_val    = loaded['best_val_loss']
            if 'scheduler_state_dict' in loaded:
                scheduler.load_state_dict(loaded['scheduler_state_dict'])
            if 'D_sharp_state_dict' in loaded:
                D_sharp.module.load_state_dict(loaded['D_sharp_state_dict'])
                D_smooth.module.load_state_dict(loaded['D_smooth_state_dict'])
            if 'opt_D_state_dict' in loaded:
                opt_D.load_state_dict(loaded['opt_D_state_dict'])

    for epoch in range(start_epoch, cfg.epochs):
        ep = epoch + 1

        img_train_sampler.set_epoch(epoch)

        cur_lr = optimizer.param_groups[0]['lr']
        if is_main:
            print(f"\n--- Epoch {ep}/{cfg.epochs}  (lr={cur_lr:.2e}) ---")

        train_stats, plot_data = train_one_epoch(
            model, D_sharp, D_smooth, img_train_loader,
            optimizer, opt_D, scaler, cfg.alpha, cfg.lambda_adv,
            device, epoch=ep, is_main=is_main
        )
        val_stats = validate_adv(
            model, D_sharp, D_smooth, img_val_loader,
            cfg.alpha, cfg.lambda_adv, device
        )

        train_stats = all_reduce_stats(train_stats)
        val_stats   = all_reduce_stats(val_stats)

        if is_main:
            plot_epoch_results(plot_data, ep, out_dir)

            scheduler.step(val_stats['total_loss'])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < cur_lr:
                print(f"LR dropped: {cur_lr:.2e} -> {new_lr:.2e}")

            csv_writer.writerow([  # type: ignore
                ep, new_lr,
                train_stats['total_loss'], train_stats['d_loss'], train_stats['g_adv_loss'],
                train_stats['ft_loss'], train_stats['grad_norm'], train_stats.get('nan_batches', 0),
                val_stats['total_loss'], val_stats['g_adv_loss'], val_stats['ft_loss'],
            ])
            csv_file.flush()  # type: ignore

            print(
                f"  train — total: {train_stats['total_loss']:.4f}  D: {train_stats['d_loss']:.4f}"
                f"  G_adv: {train_stats['g_adv_loss']:.4f}  ft: {train_stats['ft_loss']:.4f}"
            )
            print(
                f"  val   — total: {val_stats['total_loss']:.4f}  G_adv: {val_stats['g_adv_loss']:.4f}"
                f"  ft: {val_stats['ft_loss']:.4f}"
            )

            is_best = val_stats['total_loss'] < best_val
            if is_best:
                best_val = val_stats['total_loss']
                print(f"  ** new best val loss: {best_val:.6f} **")

            raw_model   = model.module
            raw_D_sharp  = D_sharp.module
            raw_D_smooth = D_smooth.module
            ckpt = {
                'epoch':                ep,
                'model_state_dict':     raw_model.state_dict(),
                'D_sharp_state_dict':   raw_D_sharp.state_dict(),
                'D_smooth_state_dict':  raw_D_smooth.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'opt_D_state_dict':     opt_D.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict':    scaler.state_dict(),
                'best_val_loss':        best_val,
                'alpha':                cfg.alpha,
                'lambda_adv':           cfg.lambda_adv,
                'learning_rate':        cfg.lr,
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{ep}_checkpoint.pth")
            if is_best:
                torch.save(ckpt, ckpt_dir / "best_checkpoint.pth")
        else:
            scheduler.step(val_stats['total_loss'])

        dist.barrier()

    if is_main:
        csv_file.close()  # type: ignore
        print(f"\nDone. Best val loss: {best_val:.6f}")
        print(f"Metrics saved to: {csv_path}")  # type: ignore

    cleanup_ddp()


if __name__ == "__main__":
    main()
