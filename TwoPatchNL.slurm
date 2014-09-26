#!/bin/bash

#SBATCH -J ImProc_4x

#SBATCH -t 20:00:00
#SBATCH -N 1 # node
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4096M


#SBATCH --mail-user=rbirger@princeton.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL



python TwoPatchModelNoLatent.py $SLURM_ARRAY_TASK_ID



# submit using sbatch
# scontrol show job 1373263
# squeue -u rbirger
# scancel 1423435
