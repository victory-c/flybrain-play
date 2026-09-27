#!/bin/bash
#SBATCH --job-name=flybench
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flybench-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flybench-%j.err
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/s/st/stevejobs/flybrain/fly_brain
/home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.bench_gpu "$@"
