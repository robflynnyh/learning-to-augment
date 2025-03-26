#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=16

echo "Task id is $SLURM_ARRAY_TASK_ID"

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

nproc
taskset -cp $$

cd ../

python generate.py --config ./configs/configs_in_paper/CM_train/CM.yaml --index $INDEX --steps 5 --repeats 18 --skip_percentage 98.0 --remove_skipped_paths
#taskset -c 0,1,2,3,4,5,6,7 python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID & taskset -c 8,9,10,11,12,13,14,15 python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID

# if [ $? -ne 0 ]; then
#     echo "taskset failed. Running on all cores instead."
#     python generate.py --config ./configs/example.yaml --index $SLURM_ARRAY_TASK_ID
# fi


