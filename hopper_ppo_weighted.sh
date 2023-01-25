#!/bin/bash
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-72:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique

source $HOME/Documents/ENV/bin/activate
module load python/3.10
module load mujoco mpi4py

SECONDS=0
python Hyperparam/weighted_ppo_tune.py --seed 189 --log_dir $SCRATCH/avg_discount/ --env 'Hopper-v4' --epochs 500 &
python Hyperparam/weighted_ppo_tune.py --seed 189 --log_dir $SCRATCH/avg_discount/ --env 'Swimmer-v4' --epochs 500 &
python Hyperparam/weighted_ppo_tune.py --seed 189 --log_dir $SCRATCH/avg_discount/ --env 'Reacher-v4' --epochs 500 &
#python Hyperparam/weighted_ppo_tune.py --seed $SLURM_ARRAY_TASK_ID --log_dir $SCRATCH/avg_discount/ --env 'Ant-v4' --epochs 500&

echo "Baseline job $seed took $SECONDS"
sleep 72h