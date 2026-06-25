#!/bin/bash -l
#SBATCH --job-name=dofa_unmix
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=V100-32GB
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --exclude=node42,node43
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# Activate the environment
module load cuda
eval "$(conda shell.bash hook)"
conda activate hsu

# Define variables for the job
N_XP=10
N_TRAIN=15
VERSION="v1"
SIZE="large"
MODEL="DOFA"
UPSAMPLER="Features_fusion"

# Execute the Python script with specific arguments
# srun python /home/ids/edabier/HSU/SS-HSU_benchmark/fm_unmixing.py --n_xp $N_XP
# srun python /home/ids/edabier/HSU/SS-HSU_benchmark/dofa_n_training.py
# srun python /home/ids/edabier/HSU/SS-HSU_benchmark/padding_training_2.py
# srun python /home/ids/edabier/HSU/SS-HSU_benchmark/dofa_15_training.py --version $VERSION --size $SIZE --model $MODEL
# srun python /home/ids/edabier/HSU/SS-HSU_benchmark/dofa_shift_15_training.py --n_xp $N_XP --n_train $N_TRAIN --version $VERSION --size $SIZE
# srun python /home/ids/edabier/HSU/SS-HSU_benchmark/train_upsamplers.py --upsampler $UPSAMPLER --model $MODEL
srun python /home/ids/edabier/HSU/SS-HSU_benchmark/train_upsamplers_ablation.py --model $MODEL --upsampler $UPSAMPLER

# Print job completion time
echo "Job finished at: $(date)"