#!/bin/bash
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem-per-cpu=1024M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-72:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique

source $HOME/Documents/ENV/bin/activate
module load python/3.10
module load mujoco mpi4py

SECONDS=0
srun --ntasks=1 python mountainCar_initial_test.py --env="Swimmer-v4" --seed=394 --log_dir $SCRATCH/avg_discount/dm_diff/ --epochs 500&
srun --ntasks=1 python mountainCar_initial_test.py --env="HalfCheetah-v4" --seed=556 --log_dir $SCRATCH/avg_discount/dm_diff/ --epochs 500&
srun --ntasks=1 python mountainCar_initial_test.py --env="Ant-v4" --seed=790 --log_dir $SCRATCH/avg_discount/dm_diff/ --epochs 500&
srun --ntasks=1 python mountainCar_initial_test.py --env="Walker2d-v4" --seed=566 --log_dir $SCRATCH/avg_discount/dm_diff/ --epochs 500&
srun --ntasks=1 python mountainCar_initial_test.py --env="MountainCarContinuous-v0" --seed=243 --log_dir $SCRATCH/avg_discount/dm_diff/ --epochs 500&

wait
echo "Baseline job $seed took $SECONDS"
sleep 72h