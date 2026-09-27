#!/bin/bash
# Serve one drink to the fly on the HPC node.  Example:
#   ./serve.sh "🧋 奶茶" --sugar 80 --caffeine 150 --ph 6.5
#   ./serve.sh "🍺 青岛" --abv 4.7 --sugar 3 --ibu 15 --co2 5 --ph 4.3 --hunger 0.6
cd /home/s/st/stevejobs/flybrain/fly_brain
exec srun -p ocf-hpc -w corruption -c 32 --mem=32G -t 20 --quiet \
  env OMP_NUM_THREADS=32 /home/s/st/stevejobs/flybrain/venv/bin/python -m runs.serve "$@"
