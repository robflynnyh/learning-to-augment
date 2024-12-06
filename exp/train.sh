#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=100GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate a100

python train.py --config ./configs/example.yaml