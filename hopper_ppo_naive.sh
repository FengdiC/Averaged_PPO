#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-168:00
#SBATCH --output=%N-%j-naive.out
#SBATCH --account=def-ashique

source $HOME/Documents/ENV/bin/activate
module load python/3.10
module load mujoco mpi4py

SECONDS=0
python Hyperparam/naive_ppo_tune.py --seed 226 --log_dir $SCRATCH/avg_discount/logs/ --env 'Hopper-v4' --epochs 1500 &
python Hyperparam/naive_ppo_tune.py --seed 347 --log_dir $SCRATCH/avg_discount/logs/ --env 'Ant-v4' --epochs 1500 &
#python Hyperparam/naive_ppo_tune.py --seed 2 --log_dir $SCRATCH/avg_discount/ --env 'Reacher-v4' --epochs 500 &

#python Hyperparam/naive_ppo_tune.py --seed $SLURM_ARRAY_TASK_ID --log_dir $SCRATCH/avg_discount/halfcheetah/ --env 'HalfCheetah-v4' --epochs 500 &
echo "Baseline job $seed took $SECONDS"
sleep 168h