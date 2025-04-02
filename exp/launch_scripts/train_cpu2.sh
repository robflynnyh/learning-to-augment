#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

#echo "Running train_freq_mask.py with config $CONFIG"

cd ../

#python train_freq_mask.py --config ./configs/configs_in_paper/DT_train/DT.yaml
python train_freq_mask.py --config ./configs/configs_in_paper/conditional_mask_lm/MLM.yaml

# example use: sbatch --export=CONFIG='./configs/example.yaml'  ./train_cpu.sh
