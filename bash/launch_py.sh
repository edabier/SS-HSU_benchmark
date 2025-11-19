#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --output=%x_%j.out      # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err       # Error file
#SBATCH -p P100
#SBATCH --nodes=1                     
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for the job
N_WORKERS=$SLURM_CPUS_PER_TASK
EPOCHS=200
BATCH_SIZE=4

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate hsu-env

# Execute the Python script with specific arguments
srun python /home/ids/edabier/HSU/SS-HSU_benchmark/bash/trainer.py 

# Retrieve and log job information
LOG_FILE="job_tracking.log"
echo "Job Tracking Log - $(date)" >> $LOG_FILE
sacct -u $USER --format=JobID,JobName,Partition,Elapsed,State >> $LOG_FILE
echo "----------------------------------------" >> $LOG_FILE

# Print job completion time
echo "Job finished at: $(date)"