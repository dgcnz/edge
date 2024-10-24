#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --job-name=try_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:10:00
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

cd $HOME/development/edge
source .venv/bin/activate

cd $HOME/development/edge/detrex
export DETECTRON2_DATASETS=$HOME/datasets
python tools/train_net.py \
    --config-file ../projects/dino_dinov2/configs/COCO/dino_dinov2_b_12ep.py \
    --num-gpus 4 \
    train.fast_dev_run.enabled=True

deactivate
module unload CUDA/12.1.1
module unload Python/3.11.3-GCCcore-12.3.0
module unload 2023