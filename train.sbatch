#!/bin/bash

#SBATCH --job-name csp   ## name that will show up in the queue
#SBATCH --output csp_%x_%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=4  ## the number of threads allocated to each task
#SBATCH --gpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --time=0-10:01:00  ## time for analysis (day-hour:min:sec)
## SBATCH --mail-user ipromanov@edu.hse.ru  ## your email address
#SBATCH --export=ALL
## SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
## SBATCH --mail-type END  ## slurm will email you when your job ends
## SBATCH --mail-type FAIL  ## slurm will email you when your job fails
## Load modules


module load Python/Anaconda_v11.2020

conda activate csp_env

srun --ntasks=1 --gpus=1 --exclusive -N1 -n1 --cpus-per-task=4 WANDB__SERVICE_WAIT=300 python train.py

wait
