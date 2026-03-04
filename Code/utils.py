import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import BSpline
import os
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import cycle
from scipy import signal
import logging
from datetime import datetime

huber = nn.HuberLoss(delta=1.0)


def setup_logging(output_dir):
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger

def plot_images_for_epoch(I_smooth, I_sharp, I_gen_sharp, I_gen_smooth, epoch, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    I_smooth_np = I_smooth.detach().cpu().float().numpy().squeeze()
    I_sharp_np = I_sharp.detach().cpu().float().numpy().squeeze()
    
    I_gen_sharp_np = I_gen_sharp.detach().cpu().float().numpy().squeeze()
    if I_gen_sharp_np.ndim == 3: I_gen_sharp_np = I_gen_sharp_np[0]
        
    I_gen_smooth_np = I_gen_smooth.detach().cpu().float().numpy().squeeze()
    if I_gen_smooth_np.ndim == 3: I_gen_smooth_np = I_gen_smooth_np[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    im0 = axes[0, 0].imshow(I_smooth_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('I_smooth (Input)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(I_sharp_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('I_sharp (Input)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[1, 0].imshow(I_gen_sharp_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('I_generated_sharp (from I_smooth)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    im3 = axes[1, 1].imshow(I_gen_smooth_np, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('I_generated_smooth (from I_sharp)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.suptitle(f'Image Reconstruction - Epoch {epoch}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = output_dir / f'epoch_{epoch:03d}_images.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_splines_for_epoch(knots_smooth, control_smooth, knots_sharp, control_sharp,
                           knots_phantom, control_phantom, target_mtf, epoch, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mtf_smooth_torch = get_torch_spline(knots_smooth, control_smooth, num_points=363)[0, 0].detach().cpu().float().numpy()
    mtf_sharp_torch = get_torch_spline(knots_sharp, control_sharp, num_points=363)[0, 0].detach().cpu().float().numpy()
    mtf_phantom_torch = get_torch_spline(knots_phantom, control_phantom, num_points=64)[0, 0].detach().cpu().float().numpy()
    k_s = knots_smooth[0].detach().cpu().float()
    c_s = control_smooth[0].detach().cpu().float()
    k_sh = knots_sharp[0].detach().cpu().float()
    c_sh = control_sharp[0].detach().cpu().float()
    k_p = knots_phantom[0].detach().cpu().float()
    c_p = control_phantom[0].detach().cpu().float()
    x_smooth_scipy, y_smooth_scipy = get_scipy_spline(k_s, c_s, num_points=363)
    x_sharp_scipy, y_sharp_scipy = get_scipy_spline(k_sh, c_sh, num_points=363)
    x_phantom_scipy, y_phantom_scipy = get_scipy_spline(k_p, c_p, num_points=64)
    target_mtf_np = target_mtf[0].detach().cpu().float().numpy()
    x_torch_363 = np.linspace(0, 1, 363)
    x_torch_64 = np.linspace(0, 1, 64)
    x_target = np.linspace(0, 1, len(target_mtf_np))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(x_torch_363, mtf_smooth_torch, 'b-', linewidth=2.5, label='Torch Spline', alpha=0.8)
    axes[0, 0].plot(x_smooth_scipy, y_smooth_scipy, 'r--', linewidth=2, label='SciPy Spline', alpha=0.7)
    axes[0, 0].set_xlabel('Normalized Frequency', fontsize=12)
    axes[0, 0].set_ylabel('MTF Value', fontsize=12)
    axes[0, 0].set_title('Smooth Image MTF', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    
    axes[0, 1].plot(x_torch_363, mtf_sharp_torch, 'b-', linewidth=2.5, label='Torch Spline', alpha=0.8)
    axes[0, 1].plot(x_sharp_scipy, y_sharp_scipy, 'r--', linewidth=2, label='SciPy Spline', alpha=0.7)
    axes[0, 1].set_xlabel('Normalized Frequency', fontsize=12)
    axes[0, 1].set_ylabel('MTF Value', fontsize=12)
    axes[0, 1].set_title('Sharp Image MTF', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 1])
    
    axes[1, 0].plot(x_torch_64, mtf_phantom_torch, 'b-', linewidth=2.5, label='Torch Spline', alpha=0.8)
    axes[1, 0].plot(x_phantom_scipy, y_phantom_scipy, 'r--', linewidth=2, label='SciPy Spline', alpha=0.7)
    axes[1, 0].set_xlabel('Normalized Frequency', fontsize=12)
    axes[1, 0].set_ylabel('MTF Value', fontsize=12)
    axes[1, 0].set_title('Predicted MTF (from Phantom)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 1])
    
    axes[1, 1].plot(x_target, target_mtf_np, 'g-', linewidth=2.5, label='Ground Truth MTF', alpha=0.8)
    axes[1, 1].plot(x_torch_64, mtf_phantom_torch, 'b--', linewidth=2, label='Predicted MTF (Torch)', alpha=0.7)
    axes[1, 1].set_xlabel('Normalized Frequency', fontsize=12)
    axes[1, 1].set_ylabel('MTF Value', fontsize=12)
    axes[1, 1].set_title('Predicted vs Ground Truth MTF', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    
    plt.suptitle(f'MTF Splines - Epoch {epoch}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = output_dir / f'epoch_{epoch:03d}_splines.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

@torch.no_grad()
def validate(model, image_loader, mtf_loader, l1_loss, alpha, device):
    model.eval()

    total_loss       = 0.0
    total_recon_loss = 0.0
    total_mtf_loss   = 0.0
    total_ft_loss    = 0.0
    num_batches      = 0

    mtf_cycle = cycle(mtf_loader)

    for batch_idx, (I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2, *_) in enumerate(image_loader):
        I_smooth_1 = I_smooth_1.to(device, non_blocking=True)
        I_sharp_1  = I_sharp_1.to(device, non_blocking=True)
        I_smooth_2 = I_smooth_2.to(device, non_blocking=True)
        I_sharp_2  = I_sharp_2.to(device, non_blocking=True)

        psd_smooth_1 = compute_psd(I_smooth_1, device='cuda').to(device, non_blocking=True)
        psd_sharp_2  = compute_psd(I_sharp_2,  device='cuda').to(device, non_blocking=True)

        smooth_knots_1, smooth_control_points_1 = model(psd_smooth_1)
        sharp_knots_2,  sharp_control_points_2  = model(psd_sharp_2)

        otf_smooth,otf_sharp = spline_to_kernel(
            smooth_knots=smooth_knots_1, smooth_control_points=smooth_control_points_1,
            sharp_knots=sharp_knots_2,   sharp_control_points=sharp_control_points_2,
            grid_size=512
        )
        filter_smooth2sharp = otf_sharp/(otf_smooth + 1e-10)
        filter_sharp2smooth = otf_smooth/(otf_sharp + 1e-10)

        I_generated_sharp, I_generated_smooth = generate_images(
            I_smooth=I_smooth_1, I_sharp=I_sharp_2,
            filter_smooth2sharp=filter_smooth2sharp,
            filter_sharp2smooth=filter_sharp2smooth,
            device=device
        )

        recon_loss = (l1_loss(I_generated_sharp,  I_sharp_1) +
                      l1_loss(I_generated_smooth, I_smooth_2)) / 2.0

        input_profiles, target_mtfs, _ = next(mtf_cycle)
        input_profiles = input_profiles.to(device, non_blocking=True)
        target_mtfs    = target_mtfs.to(device, non_blocking=True)

        knots_mtf, control_mtf = model(input_profiles)
        pred_mtf = get_torch_spline(knots_mtf, control_mtf, num_points=target_mtfs.shape[-1]).squeeze(1)
        mtf_loss = l1_loss(pred_mtf, target_mtfs)

        I_smooth_fft = compute_fft(I_smooth_1)
        I_sharp_fft = compute_fft(I_sharp_1)
        eps_A = 1e-6 * (torch.median(I_smooth_fft.real))
        eps_B = 1e-6 * torch.median(I_sharp_fft.real)
        ft_loss = huber(
            torch.log(I_smooth_fft.abs() + eps_A) - torch.log(I_sharp_fft.abs() + eps_B), 
            torch.log(otf_smooth + eps_A) - torch.log(otf_sharp + eps_B)
            )

        batch_loss = alpha * recon_loss + (1 - alpha) * mtf_loss + 0.5 * ft_loss

        total_loss       += batch_loss.item()
        total_recon_loss += recon_loss.item()
        total_mtf_loss   += mtf_loss.item()
        total_ft_loss    += ft_loss.item()
        num_batches      += 1

    return {
        'total_loss': total_loss       / max(num_batches, 1),
        'recon_loss': total_recon_loss / max(num_batches, 1),
        'mtf_loss':   total_mtf_loss   / max(num_batches, 1),
        'ft_loss':    total_ft_loss    / max(num_batches, 1),
    }

def save_checkpoint(epoch, model, optimizer, scaler, metrics, best_val_loss,
                   alpha, learning_rate, checkpoint_dir, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'best_val_loss': best_val_loss,
        'alpha': alpha,
        'learning_rate': learning_rate
    }
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)
    epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, epoch_path)
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer, scaler=None):
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
        'best_val_loss': checkpoint['best_val_loss']
    }

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def plot_training_metrics(metrics, alpha, learning_rate, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    epochs = metrics['epoch']
    
    axes[0, 0].plot(epochs, metrics['train_total_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, metrics['val_total_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Total Loss (Train vs Val)', fontweight='bold', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, metrics['train_recon_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, metrics['val_recon_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[0, 1].set_title('Reconstruction Loss (L1)', fontweight='bold', fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, metrics['train_mtf_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, metrics['val_mtf_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('MTF Loss', fontsize=12)
    axes[0, 2].set_title('MTF Loss (L1)', fontweight='bold', fontsize=14)
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, metrics['train_grad_norm'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Gradient Norm', fontsize=12)
    axes[1, 0].set_title('Gradient Norm (L2)', fontweight='bold', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    best_idx = np.argmin(metrics['val_total_loss'])
    best_epoch = epochs[best_idx]
    best_loss = metrics['val_total_loss'][best_idx]
    
    axes[1, 1].plot(epochs, metrics['val_total_loss'], 'b-', linewidth=2)
    axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', linewidth=2,
                      label=f'Best Epoch: {best_epoch}')
    axes[1, 1].scatter([best_epoch], [best_loss], color='r', s=150, zorder=5, marker='*')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation Total Loss', fontsize=12)
    axes[1, 1].set_title('Best Model', fontweight='bold', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    summary_text = f"""
Training Summary
{'='*40}

Total Epochs: {len(epochs)}
Best Epoch: {best_epoch}

Best Val Loss: {best_loss:.6f}

Final Metrics:
  Train Loss: {metrics['train_total_loss'][-1]:.6f}
  Val Loss:   {metrics['val_total_loss'][-1]:.6f}

  Train Recon: {metrics['train_recon_loss'][-1]:.6f}
  Val Recon:   {metrics['val_recon_loss'][-1]:.6f}

  Train MTF: {metrics['train_mtf_loss'][-1]:.6f}
  Val MTF:   {metrics['val_mtf_loss'][-1]:.6f}

  Grad Norm: {metrics['train_grad_norm'][-1]:.6f}

Hyperparameters:
  Alpha: {alpha}
  Learning Rate: {learning_rate}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plot_path = output_dir / 'training_metrics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_images(I_smooth, I_sharp, filter_smooth2sharp, filter_sharp2smooth, device='cuda'):
    fft_smooth = compute_fft(I_smooth, device=device)
    fft_sharp  = compute_fft(I_sharp,  device=device)

    fft_generated_sharp  = fft_smooth * filter_smooth2sharp
    fft_generated_smooth = fft_sharp  * filter_sharp2smooth

    fft_generated_sharp_unshifted  = torch.fft.ifftshift(fft_generated_sharp,  dim=(-2, -1))
    fft_generated_smooth_unshifted = torch.fft.ifftshift(fft_generated_smooth, dim=(-2, -1))

    I_generated_sharp  = torch.fft.ifft2(fft_generated_sharp_unshifted,  dim=(-2, -1)).real
    I_generated_smooth = torch.fft.ifft2(fft_generated_smooth_unshifted, dim=(-2, -1)).real

    I_generated_sharp  = torch.clamp(I_generated_sharp,  0, 1)
    I_generated_smooth = torch.clamp(I_generated_smooth, 0, 1)

    return I_generated_sharp, I_generated_smooth


def normalize(img):
    if img.ndim == 3:
        img = img.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, C, H, W = img.shape
    img_flat = img.view(B, -1)
    p_min = img_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    p_high = img_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    img_normalized = (img - p_min) / (p_high - p_min + 1e-8)
    img_normalized = torch.clamp(img_normalized, 0, 1)
    if squeeze_output:
        img_normalized = img_normalized.squeeze(0)
    
    return img_normalized

def get_scipy_spline(knots_tensor, control_tensor, degree=3, num_points=363):
    t = knots_tensor.detach().cpu().numpy().flatten()
    c = control_tensor.detach().cpu().numpy().flatten()

    spl = BSpline(t, c, k=degree, extrapolate=True)
    x_min = t[3]
    x_max = t[-4]
    x_axis = np.linspace(x_min, x_max, num_points)
    y_axis = spl(x_axis)
    
    return x_axis, y_axis

def cox_de_boor(t, k, knots, degree, orig_degree=3):
    if degree == 0:
        k_start = knots[:, k].unsqueeze(1)
        k_end = knots[:, k+1].unsqueeze(1)
        mask = (t >= k_start) & (t < k_end)
        return mask.float()

    epsilon = 1e-6

    t_left = knots[:, k].unsqueeze(1)
    t_right = knots[:, k+degree].unsqueeze(1)
    den1 = t_right - t_left
    term1 = torch.where(
        den1 > epsilon,
        ((t - t_left) / den1) * cox_de_boor(t, k, knots, degree-1, orig_degree),
        torch.zeros_like(t)
    )

    t_left2 = knots[:, k+1].unsqueeze(1)
    t_right2 = knots[:, k+degree+1].unsqueeze(1)
    den2 = t_right2 - t_left2
    term2 = torch.where(
        den2 > epsilon,
        ((t_right2 - t) / den2) * cox_de_boor(t, k+1, knots, degree-1, orig_degree),
        torch.zeros_like(t)
    )

    return term1 + term2


def get_torch_spline(knots, control_points, num_points=64):
    batch_size = knots.shape[0]
    degree = 3

    t_min = knots[:, degree].unsqueeze(1)
    t_max = knots[:, -degree-1].unsqueeze(1)
    t_range = torch.linspace(0, 1, num_points, device=knots.device).unsqueeze(0)
    t = t_min + (t_max - t_min) * t_range
    t = torch.clamp(t, min=t_min, max=t_max)
    n = control_points.shape[1]
    basis_values = []
    for i in range(n):
        try:
            basis = cox_de_boor(t, i, knots, degree)
            if torch.isnan(basis).any() or torch.isinf(basis).any():
                basis = torch.zeros_like(basis)
            basis_values.append(basis)
        except Exception:
            basis_values.append(torch.zeros(batch_size, num_points, device=knots.device))
    
    basis_matrix = torch.stack(basis_values, dim=1)
    control_expanded = control_points.unsqueeze(2)
    spline = (basis_matrix * control_expanded).sum(dim=1)
    spline = torch.clamp(spline, min=-10.0, max=10.0)
    spline = spline.unsqueeze(1)
    
    if torch.isnan(spline).any() or torch.isinf(spline).any():
        spline = torch.nan_to_num(spline, nan=0.0, posinf=1.0, neginf=0.0)
    
    return spline


def compute_psd(image, device):
    with torch.no_grad():
        batch_psd = []
        for b in range(image.shape[0]):
            slice_data = image[b, 0, :, :]
            slice_ft = torch.fft.fftshift(torch.fft.fft2(slice_data))
            slice_psd = torch.abs(slice_ft) ** 2
            slice_psd = torch.log(slice_psd + 1)
            psd_min = slice_psd.min()
            psd_max = slice_psd.max()
            slice_psd = (slice_psd - psd_min) / (psd_max - psd_min + 1e-10)
            
            batch_psd.append(slice_psd)
        psd = torch.stack(batch_psd, dim=0).unsqueeze(1)
        
    return psd

def compute_fft(image, device='cuda'):
    with torch.no_grad():
        image = image.to(device)
        if image.dim() == 4:
            image = image[:, 0, :, :]
        
        fft = torch.fft.fft2(image)
        fft_shifted = torch.fft.fftshift(fft)
    return fft_shifted

def spline_to_kernel(smooth_knots, smooth_control_points, sharp_knots, sharp_control_points, grid_size=512):
    batch_size = smooth_knots.shape[0]
    device = smooth_knots.device

    center = grid_size / 2.0
    y = torch.arange(grid_size, device=device, dtype=torch.float32) - center
    x = torch.arange(grid_size, device=device, dtype=torch.float32) - center
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    distance = torch.sqrt(x_grid**2 + y_grid**2)
    max_distance = center * np.sqrt(2)
    t = distance / max_distance
    t = torch.clamp(t, 0, 1)

    num_spline_points = 256
    smooth_spline_curve = get_torch_spline(smooth_knots, smooth_control_points, num_points=num_spline_points)
    sharp_spline_curve = get_torch_spline(sharp_knots, sharp_control_points, num_points=num_spline_points)
    
    smooth_spline_curve = smooth_spline_curve.view(batch_size, 1, 1,num_spline_points)
    sharp_spline_curve = sharp_spline_curve.view(batch_size, 1, 1,num_spline_points)

    grid_x = 2.0 * t - 1.0
    grid_y = torch.zeros_like(grid_x)
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    sampling_grid = sampling_grid.expand(batch_size, -1, -1, -1)

    otf_smooth = F.grid_sample(
        smooth_spline_curve,
        sampling_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(1)

    otf_sharp = F.grid_sample(
        sharp_spline_curve,
        sampling_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(1)

    return otf_smooth, otf_sharp

def symmetric_average_2d(psd_2d):
    B, C, H, W = psd_2d.shape
    center = W // 2

    left  = psd_2d[:, :, :, :center].flip(dims=[-1])
    right = psd_2d[:, :, :, center+1:]

    min_len = min(left.shape[-1], right.shape[-1])
    left  = left[:, :, :, :min_len]
    right = right[:, :, :, :min_len]

    averaged = (left + right) / 2.0

    dc = psd_2d[:, :, :, center].unsqueeze(-1)

    smoothed = torch.cat([averaged.flip(dims=[-1]), dc, averaged], dim=-1)
    return smoothed

def gaussian_blur_2d(x, kernel_size=21, sigma=5.0):
    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32) - pad
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0).to(x.device)
    kernel = kernel.expand(x.shape[1], 1, -1, -1)
    return F.conv2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'),
                    kernel, groups=x.shape[1])
