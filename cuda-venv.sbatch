#!/bin/bash
#SBATCH --job-name=cudavenv
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/cudavenv-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/cudavenv-%j.err
set -euo pipefail
ROOT=/home/s/st/stevejobs/flybrain
VENV=$ROOT/venv-cuda
export PIP_CACHE_DIR=$ROOT/.pip-cache
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q numpy pandas pyarrow
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu121
fi
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize()
    import time; t=time.time(); (x@x).sum().item(); torch.cuda.synchronize(); print("matmul ok", round(time.time()-t,3), "s")
PY
