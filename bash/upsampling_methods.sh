#!/bin/bash -l
#SBATCH --job-name=fm_unmix
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=V100-32GB
# SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --exclude=node42,node43
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# Activate the environment
module load cuda
eval "$(conda shell.bash hook)"
conda activate hsu-latest

# Define variables for the job
N_XP=10
N_TRAIN=15
UPSAMPLER="ConvTranspose"

srun python /home/ids/edabier/HSU/SS-HSU_benchmark/experiments/upsampling_methods.py --upsampler $UPSAMPLER --n_xp $N_XP --n_train $N_TRAIN

# Print job completion time
echo "Job finished at: $(date)"