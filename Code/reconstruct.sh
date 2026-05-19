#!/bin/bash
#SBATCH --job-name=kernel_reconstruction
#SBATCH --account=dlw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --constraint=gpul40s
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err


# Activate virtual environment if you have one
# source ~/venv/bin/activate

# Navigate to code directory
cd /home/cxv166/PhantomTesting/Code

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Run training
uv run reconstruct2d.py

echo "End time: $(date)"
