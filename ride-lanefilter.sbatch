#!/bin/bash
#SBATCH --job-name=flyfilter
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flyfilter-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flyfilter-%j.err
# Off-road CEM where the lane channel enters as ONE matched-filter signal (runs/probe.py HS response pattern)
# with one learned gain; individual lane-DN weights are held near zero. Seeded from the balance-good decoder.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/s/st/stevejobs/flybrain/fly_brain
PY=/home/s/st/stevejobs/flybrain/venv-cuda/bin/python
GAINS=${1:-hs_heading=150,hs_lane=50}
OUT=${2:-results/ride_filter.json}
echo "== $(date) cem lane-filter gains='$GAINS' -> $OUT"
$PY -m runs.ride --init-theta results/ride_lane.json --readout lane --lane-filter results/probe.json --lane hs --gains "$GAINS" \
    --offroad 3.5 --lane-penalty 0.02 --riders 48 --generations 18 --seconds 12 --sigma0 0.06 --sigma-new 0.02 --sigma-lane-filter 3.0 \
    --seed ${3:-11} --out $OUT
echo "== $(date) replay"
$PY -m runs.ride --replay $OUT --readout lane --lane-filter results/probe.json --lane hs --gains "$GAINS" --riders 16 --seconds 20 --seed 7 --trace ${OUT%.json}_trace.json
echo "== done $(date)"
