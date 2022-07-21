#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem-per-cpu 4000 # memory pool for all cores
#SBATCH -t 0-4:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres gpu:4
#SBATCH --cpu_per_task 1
#SBATCH --array 0:38

module load conda
module load cuda
conda activate aeon_env
python export_video_data.py $SLURM_ARRAY_TASK_ID

conda activate DEEPLABCUT
python DLC_run.py $SLURM_ARRAY_TASK_ID

