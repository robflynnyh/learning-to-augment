#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

cd ../

python train_vae.py --config ./configs/configs_in_paper/unconditional_mask_lm/UMLM_cpu.yaml
