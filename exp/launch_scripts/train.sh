#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=70GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

cd ../

python train_freq_mask.py --config ./configs/configs_in_paper/conditional_multistep_mask_lm/MLM.yaml
