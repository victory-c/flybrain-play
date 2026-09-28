#!/bin/bash
#SBATCH --job-name=flyroad2
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flyroad2-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flyroad2-%j.err
# CEM with "leaving the road ends the run": seeded from a decoder given as $1 (default: the lane CEM result).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/s/st/stevejobs/flybrain/fly_brain
PY=/home/s/st/stevejobs/flybrain/venv-cuda/bin/python
SEED_THETA=${1:-results/ride_lane.json}
OUT=${2:-results/ride_road2.json}
echo "== $(date) cem-offroad from $SEED_THETA -> $OUT"
$PY -m runs.ride --init-theta $SEED_THETA --lane hs --offroad 3.5 --lane-penalty 0.02 --gains hs_heading=200,hs_lane=60 --riders 48 --generations 15 --seconds 12 --sigma0 0.1 --seed 5 --out $OUT
echo "== $(date) replay"
$PY -m runs.ride --replay $OUT --lane hs --gains hs_heading=200,hs_lane=60 --riders 16 --seconds 20 --seed 7 --trace ${OUT%.json}_trace.json
echo "== done $(date)"
