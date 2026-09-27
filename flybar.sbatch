#!/bin/bash
#SBATCH --job-name=flybar
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flybar-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flybar-%j.err
# Fly Bar: run the MaleCNS v1.0 whole-CNS LIF simulation (sstamou03/fly_brain).
# The simulator is CPU torch, so we ask for cores rather than a GPU.
set -euo pipefail
ROOT=/home/s/st/stevejobs/flybrain
REPO=$ROOT/fly_brain
VENV=$ROOT/venv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PIP_CACHE_DIR=$ROOT/.pip-cache

echo "== $(date) on $(hostname), $SLURM_CPUS_PER_TASK cpus"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q numpy pandas pyarrow
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
source "$VENV/bin/activate"
python -c "import torch, numpy, pandas, pyarrow; print('torch', torch.__version__, 'threads', torch.get_num_threads())"

cd "$REPO"
STEP=${1:-all}
run() { echo; echo "===== $* ($(date +%T))"; "$@"; }

if [ "$STEP" = all ] || [ "$STEP" = build ]; then
  [ -f brain.npz ] || run python -m brain.build_brain
  run python -m tests.test_sim
fi
if [ "$STEP" = all ] || [ "$STEP" = pilot ]; then
  # quick taste: 4 drinks, 5 trials, 1 s each
  run python -m runs.bar --trials 5 --only "Κόλα,Espresso,Τσίπουρο,Νερό"
fi
if [ "$STEP" = all ] || [ "$STEP" = full ]; then
  run python -m runs.bar
  run python -m runs.hunger
  run python -m export.export_ui
  run python -m export.export_brain_cloud
  run python -m export.export_web
fi
echo "== done $(date)"
