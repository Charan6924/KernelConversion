# CT Kernel Conversion

Deep learning for converting between CT reconstruction kernels (smooth ↔ sharp) using Fourier-domain operations. The model learns to estimate MTF (Modulation Transfer Function) curves from image PSD (Power Spectral Density) and uses these to transform images between different kernel types.

## Project Structure

```
Code/
├── models/           # Neural network architectures
│   ├── SplineEstimator.py      # Primary: U-Net encoder-decoder + B-spline output
│   ├── KernelEstimator.py      # Encoder-only 256-point radial profile
│   ├── KernelEstimator2d.py    # Full 2D kernel U-Net
│   ├── filterModel.py          # FilterEstimator teacher model
│   ├── Discriminator.py        # GAN discriminator
│   ├── networks.py             # Ablation study models
│   └── stylegan_networks.py    # StyleGAN2-based encoder
├── data/             # Dataset loaders
│   ├── PSDDataset.py           # NIfTI training image pairs
│   ├── Dataset.py              # MTF ground truth loader
│   └── TestDataset.py          # Inference data loader
├── training/         # Training scripts
│   ├── FullTrainLoop.py        # Primary single-GPU training
│   ├── ddpTrain.py             # Distributed DataParallel training
│   ├── ddpTrainAdvaserial.py   # DDP + GAN adversarial training
│   ├── ddpTrainComplexKernel.py # DDP 256-pt distilled training
│   ├── ddpTrainComplexKernel2d.py # DDP 2D filter training
│   ├── train.py                # Legacy training
│   └── train_256.py            # Single-GPU 256-pt training
├── inference/       # Inference scripts
│   ├── reconstruct.py          # Primary inference with SplineEstimator
│   ├── reconstruct_256.py      # Inference with 256-pt model
│   └── reconstruct2d.py        # Inference with 2D kernel model
├── scripts/         # SLURM job launchers
│   ├── train.sh
│   ├── reconstruct.sh
│   └── ddp.sh
├── utils/
│   └── utils.py     # Core signal processing utilities
├── plots/           # Training metrics and diagnostic plots
│   ├── recon_loss.png
│   ├── ft_loss.png
│   └── ...
├── FourierPhantoms.py  # DICOM processing
└── test.py             # Quick model smoke test
```

## Setup

```bash
uv sync
```

## Training

```bash
# Single GPU
uv run Code/training/FullTrainLoop.py

# With SLURM
sbatch Code/scripts/train.sh
```

## Inference

```bash
uv run Code/inference/reconstruct.py
```

## Model Pipeline

1. **Input**: CT image slice → compute PSD (Power Spectral Density)
2. **Network**: U-Net encoder-decoder processes PSD
3. **Output**: B-spline knots and control points representing the MTF curve
4. **Conversion**: MTF curves → OTF filters → Fourier multiplication → inverse FFT

## Kernel Types

Valid reconstruction kernels: `B`, `C`, `CB`, `D`, `E`, `YA`, `YB`

- **Smooth kernels** (B, C): lower high-frequency response
- **Sharp kernels** (D, E): preserve more high-frequency detail
