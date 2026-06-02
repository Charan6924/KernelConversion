#!/bin/bash
#SBATCH --job-name=cycle_gan_train
#SBATCH --partition=cgpudlw
#SBATCH --nodelist=cgput004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6    
#SBATCH --mem=64G           
#SBATCH --time=72:00:00
#SBATCH --output=/home/cxv166/PhantomTesting/logs/%j_train.out
#SBATCH --error=/home/cxv166/PhantomTesting/logs/%j_train.err

# Navigate to code directory
cd /home/cxv166/PhantomTesting/Code

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Activate venv
source /home/cxv166/PhantomTesting/.venv/bin/activate

# Run training
python training/cut_train.py
echo "End time: $(date)"
