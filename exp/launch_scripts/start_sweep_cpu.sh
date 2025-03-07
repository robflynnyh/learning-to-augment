#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=16


module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

echo "Running sweep with eval config $EVAL_CONFIG and sweep config $SWEEP_CONFIG"

cd ../

python run_sweep.py --eval_config $EVAL_CONFIG --sweep_config $SWEEP_CONFIG

# example use: sbatch --export=EVAL_CONFIG=./configs/example.yaml,SWEEP_CONFIG=./sweep_configs/PL.yaml ./start_sweep_cpu.sh
