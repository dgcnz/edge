#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd $HOME/development/.local/edge
source .venv/bin/activate

cd $HOME/development/.local/edge/detrex
python tools/train_net.py --config-file ../projects/dino_dinov2/configs/COCO/dino_dinov2_b_12ep.py 

deactivate
module unload Python/3.11.3-GCCcore-12.3.0
module unload 2023
