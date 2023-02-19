#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-168:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique

source $HOME/Documents/ENV/bin/activate
module load python/3.10
module load mujoco mpi4py

SECONDS=0
python Hyperparam/weighted_ppo_tune.py --seed 189 --log_dir $SCRATCH/avg_discount/logs --env 'Hopper-v4' --epochs 1500 &
python Hyperparam/weighted_ppo_tune.py --seed 386 --log_dir $SCRATCH/avg_discount/logs --env 'Ant-v4' --epochs 1500 &
#python Hyperparam/weighted_ppo_tune.py --seed 189 --log_dir $SCRATCH/avg_discount/logs --env 'Reacher-v4' --epochs 500 &

#python Hyperparam/weighted_ppo_tune.py --seed $SLURM_ARRAY_TASK_ID --log_dir $SCRATCH/avg_discount/halfcheetah/ --env 'HalfCheetah-v4' --epochs 500&

echo "Baseline job $seed took $SECONDS"
sleep 168h