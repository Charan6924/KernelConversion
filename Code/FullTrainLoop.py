import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from PSDDataset import PSDDataset
from Dataset import MTFPSDDataset
from SplineEstimator import KernelEstimator
from utils import (
    generate_images, get_torch_spline, load_checkpoint,
    compute_gradient_norm, validate, compute_psd,
    spline_to_kernel, compute_fft, huber
)
from pathlib import Path
from tqdm import tqdm
from itertools import cycle
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def train_one_epoch(model, image_loader, mtf_loader, optimizer, scaler, l1_loss, alpha, device, epoch):
    model.train()

    running_loss = 0.0
    running_recon = 0.0
    running_mtf = 0.0
    running_ft = 0.0
    running_grad = 0.0
    n_batches = 0
    skipped = 0

    mtf_cycle = cycle(mtf_loader)

    for i, (I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2) in enumerate(
        tqdm(image_loader, desc="Training", unit="batch")
    ):
        I_smooth_1 = I_smooth_1.to(device, non_blocking=True)
        I_sharp_1  = I_sharp_1.to(device, non_blocking=True)
        I_smooth_2 = I_smooth_2.to(device, non_blocking=True)
        I_sharp_2  = I_sharp_2.to(device, non_blocking=True)

        input_profiles, target_mtfs, _ = next(mtf_cycle)
        input_profiles = input_profiles.to(device, non_blocking=True)
        target_mtfs    = target_mtfs.to(device, non_blocking=True)

        with torch.no_grad():
            psd_smooth = compute_psd(I_smooth_1, device='cuda').to(device, non_blocking=True)
            psd_sharp  = compute_psd(I_sharp_2,  device='cuda').to(device, non_blocking=True)
            I_smooth_fft = compute_fft(I_smooth_1)
            I_sharp_fft = compute_fft(I_sharp_1)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            smooth_knots, smooth_cp = model(psd_smooth)
            sharp_knots,  sharp_cp  = model(psd_sharp)

            filt_s2sh, filt_sh2s = spline_to_kernel(
                smooth_knots=smooth_knots,
                smooth_control_points=smooth_cp,
                sharp_knots=sharp_knots,
                sharp_control_points=sharp_cp,
                grid_size=512
            )

            I_gen_sharp, I_gen_smooth = generate_images(
                I_smooth=I_smooth_1,
                I_sharp=I_sharp_2,
                filter_smooth2sharp=filt_s2sh,
                filter_sharp2smooth=filt_sh2s,
                device=device
            )

            recon_loss = (l1_loss(I_gen_sharp, I_sharp_1) + l1_loss(I_gen_smooth, I_smooth_2)) / 2.0
            knots_mtf, cp_mtf = model(input_profiles)
            pred_mtf = get_torch_spline(knots_mtf, cp_mtf, num_points=target_mtfs.shape[-1]).squeeze(1)
            mtf_loss = l1_loss(pred_mtf, target_mtfs)
            ft_loss = huber(
                torch.log(I_smooth_fft.abs() + 1e-7) - torch.log(I_sharp_fft.abs() + 1e-7),
                torch.log(filt_s2sh + 1e-7)  # filt_s2sh is otf_smooth from spline_to_kernel
            )

            loss = ft_loss + recon_loss

        optimizer.zero_grad(set_to_none=True)

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

        if i == 0:
            print("control_scale:", model.control_scale.item())

        running_loss  += loss.item()
        running_recon += recon_loss.item()
        running_ft    += ft_loss.item()
        running_mtf   += mtf_loss.item()
        running_grad  += grad_norm
        n_batches     += 1

    if skipped > 0:
        print(f"WARNING: {skipped} batches were skipped (NaN/Inf)")

    denom = max(n_batches, 1)
    stats = {
        'total_loss': running_loss  / denom,
        'recon_loss': running_recon / denom,
        'ft_loss':    running_ft    / denom,
        'mtf_loss':   running_mtf   / denom,
        'grad_norm':  running_grad  / denom,
        'nan_batches': skipped,
    }

    # Return last batch data for plotting
    plot_data = {
        'I_gen_sharp': I_gen_sharp.detach().cpu(),
        'I_gen_smooth': I_gen_smooth.detach().cpu(),
        'I_sharp_1': I_sharp_1.detach().cpu(),
        'I_smooth_2': I_smooth_2.detach().cpu(),
        'smooth_knots': smooth_knots.detach().cpu(),
        'smooth_cp': smooth_cp.detach().cpu(),
        'sharp_knots': sharp_knots.detach().cpu(),
        'sharp_cp': sharp_cp.detach().cpu(),
        'filt_s2sh': filt_s2sh.detach().cpu(),
        'filt_sh2s': filt_sh2s.detach().cpu(),
    }
    return stats, plot_data


def plot_epoch_results(plot_data, epoch, out_dir):
    """Plot generated images, splines, and filter slices once per epoch."""
    vis_dir = out_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Epoch {epoch}', fontsize=14)

    # Row 1: Generated images vs targets
    # Generated sharp
    axes[0, 0].imshow(plot_data['I_gen_sharp'][0, 0].numpy(), cmap='gray')
    axes[0, 0].set_title('Generated Sharp')
    axes[0, 0].axis('off')

    # Target sharp
    axes[0, 1].imshow(plot_data['I_sharp_1'][0, 0].numpy(), cmap='gray')
    axes[0, 1].set_title('Target Sharp')
    axes[0, 1].axis('off')

    # Generated smooth
    axes[0, 2].imshow(plot_data['I_gen_smooth'][0, 0].numpy(), cmap='gray')
    axes[0, 2].set_title('Generated Smooth')
    axes[0, 2].axis('off')

    # Target smooth
    axes[0, 3].imshow(plot_data['I_smooth_2'][0, 0].numpy(), cmap='gray')
    axes[0, 3].set_title('Target Smooth')
    axes[0, 3].axis('off')

    # Row 2: Splines and filter slices
    # Smooth spline (MTF curve)
    smooth_spline = get_torch_spline(plot_data['smooth_knots'], plot_data['smooth_cp'], num_points=256)
    sharp_spline = get_torch_spline(plot_data['sharp_knots'], plot_data['sharp_cp'], num_points=256)

    axes[1, 0].plot(smooth_spline[0, 0].numpy(), label='Smooth MTF', color='blue')
    axes[1, 0].plot(sharp_spline[0, 0].numpy(), label='Sharp MTF', color='red')
    axes[1, 0].set_title('MTF Splines')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Response')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3)

    # Filter smooth2sharp slice at row 255
    filt_s2sh = plot_data['filt_s2sh']
    axes[1, 1].plot(filt_s2sh[0, 0, 255, :].numpy(), color='green')
    axes[1, 1].set_title('Filter S→Sh [0,0,255,:]')
    axes[1, 1].set_xlabel('Column index')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)

    # Filter sharp2smooth slice at row 255
    filt_sh2s = plot_data['filt_sh2s']
    axes[1, 2].plot(filt_sh2s[0, 0, 255, :].numpy(), color='orange')
    axes[1, 2].set_title('Filter Sh→S [0,0,255,:]')
    axes[1, 2].set_xlabel('Column index')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].grid(True, alpha=0.3)

    # 2D filter visualization (smooth2sharp)
    axes[1, 3].imshow(filt_s2sh[0, 0].numpy(), cmap='viridis')
    axes[1, 3].set_title('Filter S→Sh (2D)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(vis_dir / f'epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    IMAGE_ROOT = r"/home/cxv166/PhantomTesting/Data_Root"
    MTF_FOLDER = r"/home/cxv166/PhantomTesting/MTF_Results_Output"
    PSD_FOLDER = r"/home/cxv166/PhantomTesting/PSD_Results_Output"

    ALPHA      = 0.5
    LR         = 1e-4
    EPOCHS     = 150
    BATCH_SIZE = 32
    RESUME     = False

    SCHED_FACTOR    = 0.5
    SCHED_PATIENCE  = 5
    SCHED_MIN_LR    = 1e-7

    out_dir       = Path(f"training_output_{ALPHA}")
    ckpt_dir      = out_dir / "checkpoints"

    for d in [out_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup CSV logging
    csv_path = out_dir / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'learning_rate',
        'train_total_loss', 'train_recon_loss', 'train_ft_loss', 'train_mtf_loss', 'train_grad_norm', 'nan_batches',
        'val_total_loss', 'val_recon_loss', 'val_mtf_loss', 'val_ft_loss'
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  alpha={ALPHA}  |  lr={LR}  |  epochs={EPOCHS}")

    img_dataset = PSDDataset(root_dir=IMAGE_ROOT, preload=True)
    n_train = int(0.9 * len(img_dataset))
    img_train, img_val = random_split(
        img_dataset, [n_train, len(img_dataset) - n_train],
        generator=torch.Generator().manual_seed(42)
    )

    mtf_dataset = MTFPSDDataset(MTF_FOLDER, PSD_FOLDER, verbose=True)
    m_train = int(0.8 * len(mtf_dataset))
    mtf_train, mtf_val = random_split(
        mtf_dataset, [m_train, len(mtf_dataset) - m_train],
        generator=torch.Generator().manual_seed(42)
    )

    img_train_loader = DataLoader(img_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    img_val_loader   = DataLoader(img_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    mtf_train_loader = DataLoader(mtf_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    mtf_val_loader   = DataLoader(mtf_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Images  — train: {len(img_train)}, val: {len(img_val)}")
    print(f"MTF     — train: {len(mtf_train)},  val: {len(mtf_val)}")

    model     = KernelEstimator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR
    )
    scaler  = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    l1_loss = nn.L1Loss()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_epoch  = 0
    best_val     = float('inf')

    if RESUME:
        ckpt_path = ckpt_dir / "latest_checkpoint.pth"
        loaded = load_checkpoint(ckpt_path, model, optimizer, scaler)
        if loaded:
            start_epoch = loaded['epoch'] + 1
            best_val    = loaded['best_val_loss']
            if 'scheduler_state_dict' in loaded:
                scheduler.load_state_dict(loaded['scheduler_state_dict'])

    for epoch in range(start_epoch, EPOCHS):
        ep = epoch + 1
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {ep}/{EPOCHS}  (lr={cur_lr:.2e}) ---")

        train_stats, plot_data = train_one_epoch(
            model, img_train_loader, mtf_train_loader,
            optimizer, scaler, l1_loss, ALPHA, device, epoch=ep
        )
        val_stats = validate(
            model, img_val_loader, mtf_val_loader,
            l1_loss, ALPHA, device
        )

        # Plot once per epoch
        plot_epoch_results(plot_data, ep, out_dir)

        scheduler.step(val_stats['total_loss'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < cur_lr:
            print(f"LR dropped: {cur_lr:.2e} -> {new_lr:.2e}")

        # Write to CSV
        csv_writer.writerow([
            ep, new_lr,
            train_stats['total_loss'], train_stats['recon_loss'], train_stats['ft_loss'],
            train_stats['mtf_loss'], train_stats['grad_norm'], train_stats.get('nan_batches', 0),
            val_stats['total_loss'], val_stats['recon_loss'], val_stats['mtf_loss'], val_stats['ft_loss']
        ])
        csv_file.flush()

        print(
            f"  train — total: {train_stats['total_loss']:.4f}  recon: {train_stats['recon_loss']:.4f}"
            f"  mtf: {train_stats['mtf_loss']:.4f}"
        )
        print(
            f"  val   — total: {val_stats['total_loss']:.4f}  recon: {val_stats['recon_loss']:.4f}"
            f"  mtf: {val_stats['mtf_loss']:.4f}"
        )

        is_best = val_stats['total_loss'] < best_val
        if is_best:
            best_val = val_stats['total_loss']
            print(f"  ** new best val loss: {best_val:.6f} **")

        ckpt = {
            'epoch': ep,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict() if scaler else None,
            'best_val_loss': best_val,
            'alpha':      ALPHA,
            'learning_rate': LR,
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{ep}_checkpoint.pth")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best_checkpoint.pth")

    csv_file.close()
    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()