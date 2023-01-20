#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-72:00
#SBATCH --output=%N-%j-naive.out
#SBATCH --account=def-ashique

source $HOME/Documents/ENV/bin/activate
module load python/3.10
module load mujoco mpi4py

SECONDS=0
python Hyperparam/naive_ppo_tune.py --seed 226 --log_dir $SCRATCH/avg_discount/ --env 'Hopper-v4' --epochs 500 &
python Hyperparam/naive_ppo_tune.py --seed 226 --log_dir $SCRATCH/avg_discount/ --env 'Swimmer-v4' --epochs 500 &
python Hyperparam/naive_ppo_tune.py --seed 226 --log_dir $SCRATCH/avg_discount/ --env 'Ant-v4' --epochs 500 &
#python Hyperparam/run_mujoco_naive.py --seed $SLURM_ARRAY_TASK_ID --log_dir $SCRATCH/avg_discount/ &
echo "Baseline job $seed took $SECONDS"
sleep 72h