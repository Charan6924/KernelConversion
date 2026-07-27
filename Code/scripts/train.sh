#!/bin/bash
#SBATCH --job-name=ldm_ct_stage2
#SBATCH --partition=cgpudlw
#SBATCH --nodelist=cgput004
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
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

# Run stage-2 training (conditional LDM: smooth -> sharp CT)
python main.py --base configs/latent-diffusion/ct_smooth2sharp_f8.yaml -t --gpus 0,1,2,3

echo "End time: $(date)"
