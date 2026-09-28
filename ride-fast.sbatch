#!/bin/bash
#SBATCH --job-name=flyfast
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flyfast-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flyfast-%j.err
# Off-road CEM with the HS-driven "lane" descending neurons added to the readout (runs/probe.py found them),
# seeded from the balance-good decoder (results/ride_lane.json). $1 = sensory gain overrides ('' = default), $2 = out.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/s/st/stevejobs/flybrain/fly_brain
PY=/home/s/st/stevejobs/flybrain/venv-cuda/bin/python
GAINS=${1:-}
OUT=${2:-results/ride_lanedn.json}
echo "== $(date) cem lane-readout gains='$GAINS' -> $OUT"
$PY -m runs.ride --init-theta results/ride_lane.json --readout lane --lane hs --gains "$GAINS" --offroad 3.5 --lane-penalty 0.02 --v0 5.5 \
    --riders 48 --generations 18 --seconds 12 --sigma0 0.08 --sigma-new ${3:-0.8} --seed ${4:-6} --out $OUT
echo "== $(date) replay"
$PY -m runs.ride --replay $OUT --readout lane --lane hs --gains "$GAINS" --v0 5.5 --riders 16 --seconds 20 --seed 7 --trace ${OUT%.json}_trace.json
echo "== done $(date)"
