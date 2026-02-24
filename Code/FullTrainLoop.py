import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from PSDDataset import PSDDataset
from Dataset import MTFPSDDataset
from SplineEstimator import KernelEstimator
from utils import (
    generate_images, get_torch_spline, save_checkpoint, load_checkpoint,
    compute_gradient_norm, plot_training_metrics, validate, compute_psd,
    plot_images_for_epoch, plot_splines_for_epoch, setup_logging,
    spline_to_kernel, compute_ratios
)
from pathlib import Path
from tqdm import tqdm
from itertools import cycle
import json
import numpy as np
import logging
from datetime import datetime


def train_one_epoch(model, image_loader, mtf_loader, optimizer, scaler, l1_loss, alpha, device, logger):
    model.train()

    running_loss = 0.0
    running_recon = 0.0
    running_mtf = 0.0
    running_ft = 0.0
    running_grad = 0.0
    n_batches = 0
    skipped = 0

    mtf_cycle = cycle(mtf_loader)

    # keep track of the last batch for visualization later
    vis = {}

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
            real_s2sh, real_sh2s = compute_ratios(psd_smooth, psd_sharp)
            ft_loss = torch.abs(real_sh2s - filt_sh2s) + torch.abs(real_s2sh - filt_s2sh)
            ft_loss = torch.log(ft_loss.mean() + 1)

            
            if i == 0:
                logger.info("Filter Range Debug")
                logger.info(f"  filt_s2sh  : min={filt_s2sh.min().item():.4f}  max={filt_s2sh.max().item():.4f}  mean={filt_s2sh.mean().item():.4f}")
                logger.info(f"  filt_sh2s  : min={filt_sh2s.min().item():.4f}  max={filt_sh2s.max().item():.4f}  mean={filt_sh2s.mean().item():.4f}")
                logger.info(f"  real_s2sh  : min={real_s2sh.min().item():.4f}  max={real_s2sh.max().item():.4f}  mean={real_s2sh.mean().item():.4f}")
                logger.info(f"  real_sh2s  : min={real_sh2s.min().item():.4f}  max={real_sh2s.max().item():.4f}  mean={real_sh2s.mean().item():.4f}")
                logger.info("Loss Magnitudes (epoch 1, batch 1)")
                logger.info(f"  recon_loss : {recon_loss.item():.4f}")
                logger.info(f"  mtf_loss   : {mtf_loss.item():.4f}")
                logger.info(f"  ft_loss    : {ft_loss.item():.4f}")
                logger.info(f"  ratio ft/recon : {ft_loss.item()/recon_loss.item():.2f}x")
                logger.info(f"  ratio ft/mtf   : {ft_loss.item()/mtf_loss.item():.2f}x")
                logger.info(f"  pred_mtf  : min={pred_mtf.min().item():.4f}  max={pred_mtf.max().item():.4f}  mean={pred_mtf.mean().item():.4f}")
                logger.info(f"  target_mtf: min={target_mtfs.min().item():.4f}  max={target_mtfs.max().item():.4f}  mean={target_mtfs.mean().item():.4f}")
            

            loss = alpha * recon_loss + (1 - alpha) * mtf_loss + 0.01 * ft_loss

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

        running_loss  += loss.item()
        running_recon += recon_loss.item()
        running_ft    += ft_loss.item()
        running_mtf   += mtf_loss.item()
        running_grad  += grad_norm
        n_batches     += 1

        # save for vis
        vis = {
            'I_smooth':       I_smooth_1.detach(),
            'I_sharp':        I_sharp_2.detach(),
            'I_gen_sharp':    I_gen_sharp.detach(),
            'I_gen_smooth':   I_gen_smooth.detach(),
            'knots_smooth':   smooth_knots.detach(),
            'control_smooth': smooth_cp.detach(),
            'knots_sharp':    sharp_knots.detach(),
            'control_sharp':  sharp_cp.detach(),
            'knots_phantom':  sharp_knots.detach(),
            'control_phantom':sharp_cp.detach(),
            'target_mtfs':    target_mtfs.detach(),
        }

    if skipped > 0:
        logger.warning(f"{skipped} batches were skipped (NaN/Inf)")

    denom = max(n_batches, 1)
    stats = {
        'total_loss': running_loss  / denom,
        'recon_loss': running_recon / denom,
        'ft_loss':    running_ft    / denom,
        'mtf_loss':   running_mtf   / denom,
        'grad_norm':  running_grad  / denom,
        'nan_batches': skipped,
    }
    return stats, vis


def main():
    IMAGE_ROOT = r"D:\Charan work file\KernelEstimator\Data_Root"
    MTF_FOLDER = r"D:\Charan work file\PhantomTesting\MTF_Results_Output"
    PSD_FOLDER = r"D:\Charan work file\PhantomTesting\PSD_Results_Output"

    ALPHA      = 0.5
    LR         = 1e-4
    EPOCHS     = 150
    BATCH_SIZE = 16
    RESUME     = False

    SCHED_FACTOR    = 0.5
    SCHED_PATIENCE  = 5
    SCHED_MIN_LR    = 1e-7

    out_dir       = Path(f"training_output_{ALPHA}")
    ckpt_dir      = out_dir / "checkpoints"
    vis_img_dir   = out_dir / "visualization" / "images"
    vis_spline_dir = out_dir / "visualization" / "splines"

    for d in [out_dir, ckpt_dir, vis_img_dir, vis_spline_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  |  alpha={ALPHA}  |  lr={LR}  |  epochs={EPOCHS}")

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

    logger.info(f"Images  — train: {len(img_train)}, val: {len(img_val)}")
    logger.info(f"MTF     — train: {len(mtf_train)},  val: {len(mtf_val)}")

    model     = KernelEstimator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR
    )
    scaler  = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    l1_loss = nn.L1Loss()

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    metrics = {
        'epoch': [], 'train_total_loss': [], 'train_recon_loss': [],
        'train_ft_loss': [], 'train_mtf_loss': [], 'train_grad_norm': [],
        'val_total_loss': [], 'val_recon_loss': [], 'val_mtf_loss': [],
        'learning_rate': [], 'nan_batches': []
    }
    epoch_log    = []
    start_epoch  = 0
    best_val     = float('inf')

    if RESUME:
        ckpt_path = ckpt_dir / "latest_checkpoint.pth"
        loaded = load_checkpoint(ckpt_path, model, optimizer, scaler)
        if loaded:
            start_epoch = loaded['epoch'] + 1
            metrics     = loaded['metrics']
            best_val    = loaded['best_val_loss']
            if 'scheduler_state_dict' in loaded:
                scheduler.load_state_dict(loaded['scheduler_state_dict'])

    for epoch in range(start_epoch, EPOCHS):
        ep = epoch + 1
        cur_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\n--- Epoch {ep}/{EPOCHS}  (lr={cur_lr:.2e}) ---")

        train_stats, vis_data = train_one_epoch(
            model, img_train_loader, mtf_train_loader,
            optimizer, scaler, l1_loss, ALPHA, device, logger
        )
        val_stats = validate(
            model, img_val_loader, mtf_val_loader,
            l1_loss, ALPHA, device
        )

        scheduler.step(val_stats['total_loss'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < cur_lr:
            logger.info(f"LR dropped: {cur_lr:.2e} -> {new_lr:.2e}")

        metrics['epoch'].append(ep)
        metrics['train_total_loss'].append(train_stats['total_loss'])
        metrics['train_recon_loss'].append(train_stats['recon_loss'])
        metrics['train_ft_loss'].append(train_stats['ft_loss'])
        metrics['train_mtf_loss'].append(train_stats['mtf_loss'])
        metrics['train_grad_norm'].append(train_stats['grad_norm'])
        metrics['val_total_loss'].append(val_stats['total_loss'])
        metrics['val_recon_loss'].append(val_stats['recon_loss'])
        metrics['val_mtf_loss'].append(val_stats['mtf_loss'])
        metrics['learning_rate'].append(new_lr)
        metrics['nan_batches'].append(train_stats.get('nan_batches', 0))

        epoch_log.append({
            'epoch': ep, **train_stats,
            **{f'val_{k}': v for k, v in val_stats.items()}
        })
        with open(out_dir / "training_metrics.json", 'w') as f:
            json.dump(epoch_log, f, indent=2)

        logger.info(
            f"  train — total: {train_stats['total_loss']:.4f}  recon: {train_stats['recon_loss']:.4f}"
            f"  mtf: {train_stats['mtf_loss']:.4f}  ft: {train_stats['ft_loss']:.4f}"
        )
        logger.info(
            f"  val   — total: {val_stats['total_loss']:.4f}  recon: {val_stats['recon_loss']:.4f}"
            f"  mtf: {val_stats['mtf_loss']:.4f}"
        )
        try:
            plot_images_for_epoch(
                vis_data['I_smooth'], vis_data['I_sharp'],
                vis_data['I_gen_sharp'], vis_data['I_gen_smooth'],
                ep, vis_img_dir
            )
            plot_splines_for_epoch(
                vis_data['knots_smooth'], vis_data['control_smooth'],
                vis_data['knots_sharp'],  vis_data['control_sharp'],
                vis_data['knots_phantom'],vis_data['control_phantom'],
                vis_data['target_mtfs'],
                ep, vis_spline_dir
            )
        except Exception as e:
            logger.error(f"Visualization error (epoch {ep}): {e}")

        is_best = val_stats['total_loss'] < best_val
        if is_best:
            best_val = val_stats['total_loss']
            logger.info(f"  ** new best val loss: {best_val:.6f} **")

        ckpt = {
            'epoch': ep,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict() if scaler else None,
            'metrics':    metrics,
            'best_val_loss': best_val,
            'alpha':      ALPHA,
            'learning_rate': LR,
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{ep}_checkpoint.pth")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best_checkpoint.pth")

    logger.info(f"\nDone. Best val loss: {best_val:.6f}")
    plot_training_metrics(metrics, ALPHA, LR, out_dir)


if __name__ == "__main__":
    main()