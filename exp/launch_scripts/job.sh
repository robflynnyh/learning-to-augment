#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=16
#SBATCH --array=0-550

echo "Task id is $SLURM_ARRAY_TASK_ID"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

nproc
taskset -cp $$

cd ../

python generate.py --config ./configs/generation_CM.yaml --index $SLURM_ARRAY_TASK_ID --steps 5 --repeats 6
#taskset -c 0,1,2,3,4,5,6,7 python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID & taskset -c 8,9,10,11,12,13,14,15 python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID

# if [ $? -ne 0 ]; then
#     echo "taskset failed. Running on all cores instead."
#     python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID
# fi


