#!/bin/bash

#SBATCH -p cpu # partition (queue)
#SBATCH --mem 10000

#SBATCH -t 2-4:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

python get_occupancy_from_tracking.py
