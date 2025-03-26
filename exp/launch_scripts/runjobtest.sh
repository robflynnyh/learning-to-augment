#!/bin/bash
N=50  # Set the number of runs

for i in $(seq 0 $((N-1))); do
    sbatch --export=INDEX=$i jobtest.sh
done