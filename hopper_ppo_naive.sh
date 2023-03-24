#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-72:00
#SBATCH --output=%N-%j-naive.out
#SBATCH --account=rrg-ashique

module load python/3.10
source $HOME/Documents/ENV/bin/activate
module load mujoco mpi4py

SECONDS=0
python Hyperparam/naive_ppo_tune.py --seed 535 --log_dir $SCRATCH/avg_discount/logs --env 'HalfCheetah-v4' --epochs 500  --gamma 0.97&
python Hyperparam/naive_ppo_tune.py --seed 790 --log_dir $SCRATCH/avg_discount/logs --env 'Ant-v4' --epochs 500 --gamma 0.97&
python Hyperparam/naive_ppo_tune.py --seed 151 --log_dir $SCRATCH/avg_discount/logs/ --env 'MountainCarContinuous-v0' --epochs 250 --gamma 0.97&
python Hyperparam/naive_ppo_tune.py --seed 204 --log_dir $SCRATCH/avg_discount/logs/ --env 'Pendulum-v1' --epochs 250 --gamma 0.97&
echo "Baseline job $seed took $SECONDS"
sleep 72h