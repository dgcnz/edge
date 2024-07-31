#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=install
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=scripts/slurm_logs/slurm_output_%A.out

cd $HOME/development/edge
source .venv/bin/activate

cd $HOME/development/edge/detrex
srun pip install -e .

cd $HOME/development/edge
srun pip install -e .
deactivate