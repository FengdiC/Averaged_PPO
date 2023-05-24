#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-72:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique

source $HOME/Documents/ENV/bin/activate
module load python/3.10
module load mujoco mpi4py

SECONDS=0
python Hyperparam/weighted_ppo_tune.py --seed 394 --log_dir $SCRATCH/avg_discount/swimmer --env 'Swimmer-v4' --epochs 1250&
#python Hyperparam/weighted_ppo_tune.py --seed 790 --log_dir $SCRATCH/avg_discount/logs99 --env 'Ant-v4' --epochs 500 --gamma 0.99&
#python Hyperparam/weighted_ppo_tune.py --seed 243 --log_dir $SCRATCH/avg_discount/logs99/ --env 'MountainCarContinuous-v0' --epochs 250 --gamma 0.99&
#python Hyperparam/weighted_ppo_tune.py --seed 241 --log_dir $SCRATCH/avg_discount/logs99/ --env 'Pendulum-v1' --epochs 250 --gamma 0.99&

echo "Baseline job $seed took $SECONDS"
sleep 72h