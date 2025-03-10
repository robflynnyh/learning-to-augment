#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=16


module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

echo "Running agent with command $AGENT_COMMAND"    

cd ../

wandb agent $AGENT_COMMAND

# example use: sbatch --export=AGENT_COMMAND="wobrob101/l2_augment_sweeps/lhbmnpe7" ./start_agent_cpu.sh