#!/bin/bash

#SBATCH -p gpu # partition (queue)
#SBATCH --gres gpu:4
#SBATCH -t 0-4:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --array 0-38

module load miniconda
module load cuda/11.6
conda activate aeon_env
python export_video_data.py $SLURM_ARRAY_TASK_ID

conda activate DEEPLABCUT
python DLC_run.py $SLURM_ARRAY_TASK_ID

