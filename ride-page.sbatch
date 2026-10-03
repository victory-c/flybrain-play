#!/bin/bash
#SBATCH --job-name=flypage
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flypage-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flypage-%j.err
# Replays for the /ride/ pages, with the whole brain recorded (16 riders x 20 s, seed 7, 5.5 m/s, 5 Nm):
# $1 = decoder json (brain steers, -> results/ride_road_*), then the same brains hands-off (-> results/ride_pilot_road_*).
set -euo pipefail
cd /home/s/st/stevejobs/flybrain/fly_brain
PY=/home/s/st/stevejobs/flybrain/venv-cuda/bin/python
DEC=${1:-results/lane/cem_mu.json}
COMMON="--tau-lane-ms 300 --lane-filter results/probe.json --readout lane --lane hs --gains hs_heading=150,hs_lane=50 --v0 5.5 --gust 5 --riders 16 --seconds 20 --seed 7 --center-batch"
echo "== $(date +%T) replay $DEC"
$PY -m runs.ride --replay $DEC $COMMON --trace results/ride_road_trace.json --brain-out results/ride_road_brain.npz 2>&1 | grep -E "^replay|brain ->|Error"
echo "== $(date +%T) hands-off"
$PY -m runs.ride --open-loop $COMMON --trace results/ride_pilot_road_trace.json --brain-out results/ride_pilot_road_brain.npz 2>&1 | grep -E "^open loop|brain ->|Error"
echo "== done $(date)"
