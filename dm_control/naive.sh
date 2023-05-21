#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=1-900

module load python/3.10
source $HOME/Documents/ENV/bin/activate
module load mujoco mpi4py

SECONDS=0
python Reacher_naive.py --seed $SLURM_ARRAY_TASK_ID  --type "catch" --env "ball_in_cup" --log_dir $SCRATCH/avg_discount/dm_control/ --epochs 500&
#python Reacher_naive.py --seed $SLURM_ARRAY_TASK_ID  --type "hard" --env "reacher" --log_dir $SCRATCH/avg_discount/dm_control/ --epochs 500&

echo "Baseline job $seed took $SECONDS"
sleep 72h