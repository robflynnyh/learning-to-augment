#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-50

echo "Task id is $SLURM_ARRAY_TASK_ID"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

nproc
taskset -cp $$

cd ../

python generate_search.py --config ./configs/configs_in_paper/multistep_FM_ranker/generation.yaml --split 'train' --index $SLURM_ARRAY_TASK_ID --steps 55
#taskset -c 0,1,2,3,4,5,6,7 python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID & taskset -c 8,9,10,11,12,13,14,15 python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID

# if [ $? -ne 0 ]; then
#     echo "taskset failed. Running on all cores instead."
#     python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID
# fi


