#!/bin/bash -l
#SBATCH --job-name=fm_unmixing
#SBATCH --output=%x_%j.out      # %x for job name, %j for job ID
#SBATCH --error=%x_%j.err
#SBATCH -p V100
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --exclude=node42,node43
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate hsu-env

# Define variables for the job
N_XP=5

# Execute the Python script with specific arguments
# srun python /home/ids/edabier/HSU/SS-HSU_benchmark/fm_unmixing.py --n_xp $N_XP
srun python /home/ids/edabier/HSU/SS-HSU_benchmark/dofa_n_training.py

# Retrieve and log job information
LOG_FILE="job_tracking.log"
echo "Job Tracking Log - $(date)" >> $LOG_FILE
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,State >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Print job completion time
echo "Job finished at: $(date)"