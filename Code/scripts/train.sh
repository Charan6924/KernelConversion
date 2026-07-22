#!/bin/bash
#SBATCH --job-name=ldm_ct_train
#SBATCH --partition=cgpudlw
#SBATCH --nodelist=cgput004
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=150G
#SBATCH --time=72:00:00
#SBATCH --output=/home/cxv166/PhantomTesting/logs/%j_train.out
#SBATCH --error=/home/cxv166/PhantomTesting/logs/%j_train.err

# Activate virtual environment
source /home/cxv166/PhantomTesting/.venv/bin/activate
export PYTHONPATH="/home/cxv166/PhantomTesting/Code/taming-transformers:$PYTHONPATH"
module load CUDA/12.8.0
module load GCC/12.3.0

# Navigate to code directory
cd /home/cxv166/PhantomTesting/Code/latent-diffusion

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_IB_DISABLE=1

# Run training (PyTorch Lightning DDP with 2 GPUs, launched directly — no torchrun)
CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/autoencoder/ct_autoencoder_f8.yaml -t --gpus 0,1

echo "End time: $(date)"
