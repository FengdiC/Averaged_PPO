#!/bin/bash
#SBATCH --cpus-per-task=2  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j-naive.out
#SBATCH --account=def-ashique
#SBATCH --array=0-9

source $HOME/Documents/ENV/bin/activate
module load python/3.10
module load mujoco mpi4py

SECONDS=0
python spinningup/Hyperparam/run_mujoco_naive.py --seed $SLURM_ARRAY_TASK_ID --log_dir $SCRATCH/avg_discount/ &

echo "Baseline job $seed took $SECONDS"
sleep 12h