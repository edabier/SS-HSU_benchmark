#!/bin/bash -l
#SBATCH --job-name=fill_table
#SBATCH --output=%x_%j.out      # %x for job name, %j for job ID
#SBATCH --error=%x_%j.err
#SBATCH -p V100-32GB
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
N_WORKERS=$SLURM_CPUS_PER_TASK
BATCH_SIZE=1
PATCH_SIZE=5
LR=0.00001 #0.003
EPOCHS=3000 #320
N_XP=5

# Execute the Python script with specific arguments
srun python /home/ids/edabier/HSU/SS-HSU_benchmark/fill_table.py --batch_size $BATCH_SIZE --patch_size $PATCH_SIZE --lr $LR --epochs $EPOCHS --n_xp $N_XP # --num_workers $N_WORKERS

# Retrieve and log job information
LOG_FILE="job_tracking.log"
echo "Job Tracking Log - $(date)" >> $LOG_FILE
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,State >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Print job completion time
echo "Job finished at: $(date)"