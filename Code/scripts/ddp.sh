#!/bin/bash
#SBATCH --job-name=adv_train
#SBATCH --partition=cgpudlw
#SBATCH --nodelist=cgput004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6    # can keep this
#SBATCH --mem=120G            # scale down from 120G to be safe
#SBATCH --time=72:00:00
#SBATCH --output=/home/cxv166/PhantomTesting/logs/%j_train.out
#SBATCH --error=/home/cxv166/PhantomTesting/logs/%j_train.err

source /home/cxv166/PhantomTesting/.venv/bin/activate

mkdir -p /home/cxv166/PhantomTesting/logs

cd /home/cxv166/PhantomTesting/Code

uv run training/train.py
