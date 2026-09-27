#!/bin/bash
#SBATCH --job-name=flydash
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flydash-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flydash-%j.err
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/s/st/stevejobs/flybrain/fly_brain
/home/s/st/stevejobs/flybrain/venv/bin/python -m export.export_dashboard "$@"
