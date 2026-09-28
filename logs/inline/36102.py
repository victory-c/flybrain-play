# Slurm job 36102: inline Python sent to srun on stdin, recovered from the Claude session transcript.
# timeout 300 srun -p ocf-hpc -w corruption -c 2 --mem=6G -t 5 --quiet /home/s/st/stevejobs/flybrain/venv/bin/python -
# (run from fly_brain/)
import pandas as pd, torch
from bike.senses import Senses
from bike.dynamics import Peloton
meta = pd.read_parquet('brain_meta.parquet')
s = Senses(meta, lane='hs'); b = Peloton(3); b.reset(4.0, 0.0)
b.state[:,1] = torch.tensor([2.0, -2.0, 0.0]); b.state[:,2] = torch.tensor([0.0, 0.0, 0.3])
P = s.population_rates(b.state, torch.zeros(3))
print(pd.DataFrame(P.numpy().round(1), columns=s.names)[['HS_L','HS_R']].T)
