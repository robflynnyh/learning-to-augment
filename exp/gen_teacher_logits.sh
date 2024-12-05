#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=16
#SBATCH --array=0-650

echo "Task id is $SLURM_ARRAY_TASK_ID"

module load Anaconda3/2022.10
source activate a100

nproc
taskset -cp $$

python generate_teacher_logits.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID


