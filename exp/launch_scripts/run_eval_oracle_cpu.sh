#!/bin/bash
#SBATCH --time=90:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=8


module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

echo "Running oracle_eval.py with config $CONFIG"

cd ../

python oracle_eval.py --config $CONFIG --indexes 1

# example use: sbatch --export=CONFIG='./configs/example.yaml'  ./run_eval_cpu.sh