"""Latent Diffusion inference for CT kernel conversion (smooth → sharp).

Usage:
    # Run on paired volumes in testA/ (smooth) and testB/ (sharp)
    uv run python Code/inference/reconstruct_diffusion.py \
        --ckpt logs/2026-07-28T12-00-00_ct_smooth2sharp_f8/checkpoints/last.ckpt \
        --config Code/latent-diffusion/configs/latent-diffusion/ct_smooth2sharp_f8.yaml \
        --data_root /path/to/Data_Root \
        --output_dir /path/to/output

    # Run on a single volume pair
    uv run python Code/inference/reconstruct_diffusion.py \
        --ckpt ... --config ... \
        --input_smooth /path/to/smooth.nii.gz \
        --input_sharp /path/to/sharp.nii.gz \
        --output_dir /path/to/output

Note: The model was trained for smooth → sharp conversion only.
For the reverse direction (sharp → smooth), see the --reverse flag.
"""

import os
import sys
import argparse
import glob
import numpy as np
import nibabel as nib
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latent-diffusion"))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


# ── constants ────────────────────────────────────────────────────────────────

HU_MIN, HU_MAX = -1000, 3000


# ── normalisation helpers (match training) ───────────────────────────────────


def normalize(img):
    """HU clip [-1000, 3000] → [0, 1] → [-1, 1]"""
    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    img = img * 2.0 - 1.0
    return img.astype(np.float32)


def denormalize(img):
    """[-1, 1] → [0, 1] → HU [-1000, 3000]"""
    img = (img + 1.0) / 2.0
    img = img * (HU_MAX - HU_MIN) + HU_MIN
    return np.clip(img, HU_MIN, HU_MAX).astype(np.float32)


# ── model loading ────────────────────────────────────────────────────────────


def load_model(config_path, ckpt_path, device):
    """Instantiate LatentDiffusion from config, load checkpoint, freeze
    both first-stage (VQModelInterface) and cond-stage encoders."""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {[k for k in missing if not k.startswith('loss.')][:5]}...")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, config


# ── data helpers ─────────────────────────────────────────────────────────────


def extract_kernel_name(filename):
    if "_filter_" in filename:
        return filename.split("_filter_")[1].split(".")[0]
    return "unknown"


def find_volume_pairs(root_dir, smooth_subdir="testA", sharp_subdir="testB"):
    smooth_dir = os.path.join(root_dir, smooth_subdir)
    sharp_dir = os.path.join(root_dir, sharp_subdir)
    smooth_files = sorted(f for f in os.listdir(smooth_dir) if f.endswith((".nii", ".nii.gz")))
    sharp_files = sorted(f for f in os.listdir(sharp_dir) if f.endswith((".nii", ".nii.gz")))

    sharp_dict = {
        (f.split("_filter_")[0] if "_filter_" in f else f.split(".")[0]): f
        for f in sharp_files
    }
    pairs = []
    for sf in smooth_files:
        base_id = sf.split("_filter_")[0] if "_filter_" in sf else sf.split(".")[0]
        if base_id in sharp_dict:
            pairs.append((sf, sharp_dict[base_id]))
    return pairs, smooth_dir, sharp_dir


def load_volume(path):
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32), nii.affine, nii.header


# ── latent-space encode / decode ─────────────────────────────────────────────


@torch.no_grad()
def encode_to_latent(model, x, use_cond_stage=False):
    """Encode pixel-space image (B, 1, H, W) [-1, 1] → latent (B, 4, 64, 64).

    When use_cond_stage=True, uses the cond_stage_model (VQModelInterface.encode
    which returns pre-quantization features). Otherwise uses first_stage_model
    (also VQModelInterface, identical interface).
    """
    enc = model.cond_stage_model if use_cond_stage else model.first_stage_model
    z = enc.encode(x.to(model.device))
    if not isinstance(z, torch.Tensor):
        z = z.mode() if hasattr(z, "mode") else z.sample()
    return z * model.scale_factor


@torch.no_grad()
def decode_from_latent(model, z):
    """Decode latent (B, 4, 64, 64) → pixel-space (B, 1, 512, 512) [-1, 1]."""
    z = 1.0 / model.scale_factor * z
    dec = model.first_stage_model.decode(z.to(model.device))
    return dec.float()


# ── sampling ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def sample_ddim(model, cond, ddim_steps=50, ddim_eta=0.0):
    """Run DDIM sampling conditioned on `cond` → predicted latent.

    cond: (B, 4, 64, 64) conditioning latent
    Returns: (B, 4, 64, 64) denoised latent
    """
    ddim_sampler = DDIMSampler(model)
    shape = (model.channels, model.image_size, model.image_size)
    samples, _ = ddim_sampler.sample(
        S=ddim_steps,
        batch_size=cond.shape[0],
        shape=shape,
        conditioning=cond,
        eta=ddim_eta,
        verbose=False,
    )
    return samples


# ── volume processing ────────────────────────────────────────────────────────


def process_smooth_to_sharp(smooth_vol, model, device, ddim_steps=50, ddim_eta=0.0):
    """Process a smooth volume slice-by-slice → generated sharp volume.

    Returns generated sharp volume in HU space.
    """
    n_slices = smooth_vol.shape[2]
    start = int(n_slices * 0.1)
    end = int(n_slices * 0.9)

    vol_out = np.full_like(smooth_vol, HU_MIN, dtype=np.float32)

    for k in tqdm(range(start, end), desc="  Slices", leave=False):
        slice_1ch = normalize(smooth_vol[:, :, k].copy())    # (H, W)
        x = torch.from_numpy(slice_1ch).float()[None, None, ...].to(device)  # (1, 1, H, W)

        c = encode_to_latent(model, x, use_cond_stage=True)  # smooth → latent
        z = sample_ddim(model, c, ddim_steps, ddim_eta)       # denoise → sharp latent
        out = decode_from_latent(model, z)                    # latent → pixel
        vol_out[:, :, k] = denormalize(out.cpu().numpy().squeeze())

    return vol_out


def process_sharp_to_smooth(sharp_vol, model, device, ddim_steps=50, ddim_eta=0.0):
    """Attempt sharp → smooth conversion using the model in reverse.

    The model was trained smooth→sharp, so this is an approximation.
    It conditions on the sharp latent and samples — the prior bias toward
    sharpness means results will be less smooth than desired.
    """
    n_slices = sharp_vol.shape[2]
    start = int(n_slices * 0.1)
    end = int(n_slices * 0.9)

    vol_out = np.full_like(sharp_vol, HU_MIN, dtype=np.float32)

    for k in tqdm(range(start, end), desc="  Slices (reverse)", leave=False):
        slice_1ch = normalize(sharp_vol[:, :, k].copy())
        x = torch.from_numpy(slice_1ch).float()[None, None, ...].to(device)

        # Use sharp as conditioning — this is *not* the trained direction
        c = encode_to_latent(model, x, use_cond_stage=True)
        z = sample_ddim(model, c, ddim_steps, ddim_eta)
        out = decode_from_latent(model, z)
        vol_out[:, :, k] = denormalize(out.cpu().numpy().squeeze())

    return vol_out


# ── output ────────────────────────────────────────────────────────────────────


def save_results(results, volume_id, smooth_file, sharp_file,
                 smooth_affine, sharp_affine,
                 smooth_header, sharp_header,
                 output_dir):
    os.makedirs(output_dir, exist_ok=True)

    sm_kernel = extract_kernel_name(smooth_file)
    sh_kernel = extract_kernel_name(sharp_file)

    if "sharp_from_smooth" in results:
        sub = os.path.join(output_dir, "testB_fake")
        os.makedirs(sub, exist_ok=True)
        name = f"{volume_id}_{sm_kernel}_to_{sh_kernel}.nii.gz"
        nib.save(nib.Nifti1Image(results["sharp_from_smooth"], sharp_affine, sharp_header),
                 os.path.join(sub, name))
        print(f"  Smooth→Sharp: {name}")

    if "smooth_from_sharp" in results:
        sub = os.path.join(output_dir, "testA_fake")
        os.makedirs(sub, exist_ok=True)
        name = f"{volume_id}_{sh_kernel}_to_{sm_kernel}.nii.gz"
        nib.save(nib.Nifti1Image(results["smooth_from_sharp"], smooth_affine, smooth_header),
                 os.path.join(sub, name))
        print(f"  Sharp→Smooth: {name}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Latent Diffusion inference for CT kernel conversion"
    )
    parser.add_argument("--ckpt", required=True,
                        help="Path to diffusion model checkpoint (.ckpt)")
    parser.add_argument("--config", required=True,
                        help="Path to ct_smooth2sharp_f8.yaml")
    parser.add_argument("--data_root", default=None,
                        help="Root dir containing testA/ (smooth) and testB/ (sharp)")
    parser.add_argument("--input_smooth", default=None,
                        help="Single smooth volume file")
    parser.add_argument("--input_sharp", default=None,
                        help="Single sharp volume file (paired with --input_smooth)")
    parser.add_argument("--output_dir", default="reconstructions_diffusion",
                        help="Output directory")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="DDIM sampling steps (default: 50, try 100–200 for quality)")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM stochasticity (0 = deterministic, default: 0)")
    parser.add_argument("--reverse", action="store_true",
                        help="Also attempt sharp→smooth (model wasn't trained for this)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── load ──────────────────────────────────────────────────────────────
    print(f"Loading model from {args.ckpt} ...")
    model, config = load_model(args.config, args.ckpt, device)
    print(f"  Latent diffusion: {model.channels}ch @ {model.image_size}×{model.image_size}")
    print(f"  Conditioning: {model.conditioning_key}")
    print(f"  First stage ckpt: {config.model.params.first_stage_config.params.get('ckpt_path', 'N/A')}")
    print(f"  Cond stage ckpt:  {config.model.params.cond_stage_config.params.get('ckpt_path', 'N/A')}")
    model.eval()

    # ── data ──────────────────────────────────────────────────────────────
    volume_pairs = []
    if args.input_smooth and args.input_sharp:
        volume_pairs.append((args.input_smooth, args.input_sharp))
    elif args.data_root:
        pairs, sd, shd = find_volume_pairs(args.data_root)
        volume_pairs = [(os.path.join(sd, s), os.path.join(shd, h)) for s, h in pairs]
        print(f"Found {len(volume_pairs)} volume pairs")
    else:
        parser.error("Provide either --data_root or both --input_smooth and --input_sharp")

    # ── inference ─────────────────────────────────────────────────────────
    for smooth_path, sharp_path in volume_pairs:
        vid = os.path.basename(smooth_path).split("_filter_")[0]
        print(f"\n[{vid}]")
        smooth_vol, sm_aff, sm_hdr = load_volume(smooth_path)
        sharp_vol, sh_aff, sh_hdr = load_volume(sharp_path)

        results = {}

        # Trained direction
        print("  Smooth → Sharp (trained direction):")
        results["sharp_from_smooth"] = process_smooth_to_sharp(
            smooth_vol, model, device, args.ddim_steps, args.ddim_eta)

        # Reverse direction (optional, experimental)
        if args.reverse:
            print("  Sharp → Smooth (experimental — not trained for this):")
            results["smooth_from_sharp"] = process_sharp_to_smooth(
                sharp_vol, model, device, args.ddim_steps, args.ddim_eta)

        save_results(results, vid,
                     os.path.basename(smooth_path), os.path.basename(sharp_path),
                     sm_aff, sh_aff, sm_hdr, sh_hdr, args.output_dir)

    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
