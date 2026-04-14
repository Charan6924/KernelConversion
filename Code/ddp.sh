#!/bin/bash
#SBATCH --job-name=adv_train
#SBATCH --partition=cgpudlw
#SBATCH --nodelist=cgput004
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=/home/cxv166/PhantomTesting/logs/%j_train.out
#SBATCH --error=/home/cxv166/PhantomTesting/logs/%j_train.err

source /home/cxv166/PhantomTesting/.venv/bin/activate

mkdir -p /home/cxv166/PhantomTesting/logs

cd /home/cxv166/PhantomTesting/Code

torchrun \
    --standalone \
    --nproc_per_node=4 \
    ddpTrainAdvaserial.py
