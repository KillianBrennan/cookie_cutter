#!/bin/bash -l
#SBATCH --job-name="future_cookies"
#SBATCH --account="pr133"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=killian.brennan@env.ethz.ch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

srun -A pr133 -C gpu cut_future.sh