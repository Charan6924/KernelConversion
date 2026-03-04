# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CT Kernel Conversion - A deep learning project for converting between CT reconstruction kernels (smooth ↔ sharp) using Fourier-domain operations. The model learns to estimate MTF (Modulation Transfer Function) curves from image PSD (Power Spectral Density) and uses these to transform images between different kernel types.

## Commands

### Environment Setup
```bash
uv sync                    # Install dependencies
```

### Training (HPC with SLURM)
```bash
sbatch Code/train.sh       # Submit training job to GPU cluster
uv run Code/FullTrainLoop.py  # Run training directly
```

### Inference
```bash
uv run Code/main.py        # Run inference with trained model
```

## Architecture

### Model Pipeline
1. **Input**: CT image slice → compute PSD (Power Spectral Density)
2. **Network**: U-Net encoder-decoder (`KernelEstimator`) processes PSD
3. **Output**: B-spline knots and control points representing the MTF curve
4. **Conversion**: MTF curves → OTF filters → Fourier multiplication → inverse FFT

### Key Components

**`SplineEstimator.py`** - Neural network architecture
- `KernelEstimator`: U-Net that outputs 16 values (10 control points + 6 knot parameters)
- `FixedSplineLayer`: Converts raw knot parameters into valid B-spline knot vector

**`utils.py`** - Core signal processing functions
- `compute_psd()`: Image → log-normalized power spectral density
- `compute_fft()`: Image → centered Fourier transform
- `get_torch_spline()`: Knots + control points → spline curve (Cox-de Boor algorithm)
- `spline_to_kernel()`: Spline curves → 2D radial OTF filters
- `generate_images()`: Apply OTF filter to convert images

**`PSDDataset.py`** - Training data loader for image pairs
- Loads NIfTI volumes from `trainA` (smooth) and `trainB` (sharp) directories
- Extracts kernel type from filename (`_filter_B`, `_filter_C`, etc.)
- Supports volume preloading for faster training

**`Dataset.py`** - `MTFPSDDataset` for MTF ground truth
- Pairs MTF `.mat` files with PSD `.npy` files
- Used for supervised MTF prediction loss

### Kernel Types
Valid reconstruction kernels: `B`, `C`, `CB`, `D`, `E`, `YA`, `YB`
- Smooth kernels (e.g., B, C) have lower high-frequency response
- Sharp kernels (e.g., D, E) preserve more high-frequency detail

### Training Losses
- **Reconstruction loss**: L1 between generated and target images
- **MTF loss**: L1 between predicted and ground truth MTF curves
- **FT loss**: Huber loss on log Fourier magnitude ratios

## Data Format
- Images: NIfTI (`.nii.gz`) CT volumes, normalized to [-1000, 3000] HU → [0, 1]
- MTF data: MATLAB `.mat` files with `results.mtfVal` field
- PSD data: NumPy `.npy` arrays (2D)

## Output Structure
Training outputs to `training_output_{alpha}/`:
- `checkpoints/`: Model weights (`.pth`)
- `visualization/images/`: Reconstruction comparisons per epoch
- `visualization/splines/`: MTF curve plots per epoch
- `training_metrics.json`: Loss history
