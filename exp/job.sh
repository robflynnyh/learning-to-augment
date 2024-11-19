#!/bin/bash
#SBATCH --time=07:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --array=0-550

echo "Task id is $SLURM_ARRAY_TASK_ID"

module load Anaconda3/2022.10
source activate a100

python train_test.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID


