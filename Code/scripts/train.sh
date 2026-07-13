#!/bin/bash
#SBATCH --job-name=cycle_gan_train
#SBATCH --partition=cgpudlw
#SBATCH --nodelist=cgput004
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6    
#SBATCH --mem=64G           
#SBATCH --time=72:00:00
#SBATCH --output=/home/cxv166/PhantomTesting/logs/%j_train.out
#SBATCH --error=/home/cxv166/PhantomTesting/logs/%j_train.err

# Activate virtual environment 
source /home/cxv166/PhantomTesting/.venv/bin/activate
module load CUDA/12.8.0 
module load GCC/12.3.0

# Navigate to code directory
cd /home/cxv166/PhantomTesting/Code/SynDiff

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Run training (DDP with 2 GPUs)
uv run train.py \
  --input_path /mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/ \
  --output_path /mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/train_output_diffusion/ \
  --exp my_experiment \
  --batch_size 1 \
  --image_size 512 \
  --num_epoch 500 \
  --num_process_per_node 2 \
  --contrast1 T1 --contrast2 T2 \
  --use_ema \
  --save_content \
  --save_content_every 10 \
  --save_ckpt_every 10

echo "End time: $(date)"
