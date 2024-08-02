#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=install
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

cd $HOME/development/.local/edge
source .venv/bin/activate

cd $HOME/development/.local/edge/detrex/detectron2
srun pip install -e .

cd $HOME/development/.local/edge/detrex
srun pip install -e .
deactivate

module unload CUDA/12.1.1
module unload Python/3.11.3-GCCcore-12.3.0
module unload 2023