#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH --gres gpu:10
#SBATCH --mem 64000

#SBATCH -t 2-4:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --array 0-26

module load miniconda
module load cuda/11.6
conda activate DEEPLABCUT
python DLC_run.py $SLURM_ARRAY_TASK_ID

