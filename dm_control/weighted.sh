#!/bin/bash
#SBATCH --cpus-per-task=2  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-72:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=1-250

module load python/3.10
source $HOME/Documents/ENV/bin/activate
module load mujoco mpi4py
module load dm_control

SECONDS=0
python Reacher_weighted.py --seed $SLURM_ARRAY_TASK_ID  --type "easy" --env "reacher" --log_dir $SCRATCH/avg_discount/dm_control/ --epochs 500&
python Reacher_weighted.py --seed $SLURM_ARRAY_TASK_ID  --type "hard" --env "reacher" --log_dir $SCRATCH/avg_discount/dm_control/ --epochs 500&

echo "Baseline job $seed took $SECONDS"
sleep 72h