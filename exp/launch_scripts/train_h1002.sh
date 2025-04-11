#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

ulimit -n 65536

cd ..
python train_freq_mask.py --config ./configs/configs_in_paper/multistep_FM_ranker/MLM2.yaml
