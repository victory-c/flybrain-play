#!/bin/bash
#SBATCH --job-name=flyslow
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flyslow-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flyslow-%j.err
# Lane keeping at road-bike speed: 5.5 m/s (self-stable), 5 Nm gusts. Hands-off leaves the 7 m road in ~5 s,
# so staying on it is the fly's steering. $1 = seed decoder json, $2 = out, $3 = sigma0, $4 = sigma-new, $5 = rng seed.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/s/st/stevejobs/flybrain/fly_brain
PY=/home/s/st/stevejobs/flybrain/venv-cuda/bin/python
SEEDT=$1; OUT=$2; S0=${3:-0.08}; SN=${4:-0.25}; RS=${5:-20}
COMMON="--tau-lane-ms 300 --lane-filter results/probe.json --sigma-lane-filter 1.0 --readout lane --lane hs --gains hs_heading=150,hs_lane=50 --v0 5.5 --gust 5"
echo "== $(date) cem easy from $SEEDT -> $OUT (sigma0 $S0, sigma-new $SN)"
$PY -m runs.ride --init-theta $SEEDT $COMMON --offroad 3.5 --lane-penalty 0.02 --riders 48 --generations 18 --seconds 15 --sigma0 $S0 --sigma-new $SN --seed $RS --out $OUT
echo "== $(date) replay"
$PY -m runs.ride --replay $OUT $COMMON --riders 16 --seconds 30 --seed 7 --trace ${OUT%.json}_trace.json
echo "== done $(date)"
