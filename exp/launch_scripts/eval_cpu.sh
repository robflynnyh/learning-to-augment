#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

cd ../

CONFIG_FILE=$1


python eval.py --config $CONFIG_FILE

#./configs/conditional_freq_mask.yaml
