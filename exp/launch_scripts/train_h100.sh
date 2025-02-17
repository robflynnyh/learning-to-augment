#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=150GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=32

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

python train.py --config ./configs/example.yaml
