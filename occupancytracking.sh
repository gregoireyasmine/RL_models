#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH --gres gpu:6
#SBATCH --mem 64000
#SBATCH -t 2-4:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --array 0-18


module load miniconda
python get_occupancy_from_tracking.py $SLURM_ARRAY_TASK_ID
