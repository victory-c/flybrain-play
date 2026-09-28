#!/bin/bash
#SBATCH --job-name=flysweep
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flysweep-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flysweep-%j.err
# Lane-gain sweep: the slow-lane decoder with only the lane-filter gain changed, 48 riders x 15 s, same gusts,
# leaving the road ends a run. "upright" in each summary = time upright AND on the road.
set -euo pipefail
cd /home/s/st/stevejobs/flybrain/fly_brain
PY=/home/s/st/stevejobs/flybrain/venv-cuda/bin/python
COMMON="--tau-lane-ms 300 --lane-filter results/probe.json --readout lane --lane hs --gains hs_heading=150,hs_lane=50 --v0 5.5 --gust 5 --offroad 3.5 --riders 48 --seconds 15 --seed 31"
for f in "$@"; do
  echo "== $(date +%T) $f"
  $PY -m runs.ride --replay results/sweep/$f.json $COMMON --trace results/sweep/$f.trace.json 2>&1 | grep -E "^replay"
done
echo "== done $(date)"
