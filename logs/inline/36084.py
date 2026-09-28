# Slurm job 36084: inline Python sent to srun on stdin, recovered from the Claude session transcript.
# timeout 600 srun -p ocf-hpc -w corruption -c 4 --mem=8G -t 10 --quiet ../venv/bin/python -
# (run from fly_brain/)
import pandas as pd, torch, numpy as np
from bike.senses import Senses
from bike.readout import Readout, Decoder
from bike.dynamics import Peloton
meta = pd.read_parquet('brain_meta.parquet')
s = Senses(meta); print('senses :', s.describe())
r = Readout(meta); print('readout:', r.describe()); print('features', r.n_features, 'neurons', len(r.idx))
d = Decoder(r); print('decoder dim', d.D)
b = Peloton(3); b.reset(4.0, 2.0)
b.state[:,6] = torch.tensor([0.5,-0.5,0.0]); b.state[:,5] = torch.tensor([0.1,-0.1,0.0]); b.state[:,1] = torch.tensor([0,0,2.0])
P = s.population_rates(b.state, torch.tensor([0.2,-0.2,0.0]))
print(pd.DataFrame(P.numpy().round(1), columns=s.names).T)
rates = s.rates(b.state, torch.tensor([0.2,-0.2,0.0])); print('rates', rates.shape, 'mean Hz', rates.mean().item(), 'sum input spikes/s per brain', rates.sum(1).tolist())
counts = torch.randint(0, 3, (3, len(r.idx)), dtype=torch.int32)
f = r.features(counts, 10.0); print('features', f.shape, f[0, :6])
print('gf', r.gf_spikes(counts))
theta = torch.tensor(d.prior_mean())[None].repeat(3,1); print('act', [x.tolist() for x in d.act(theta, f)])
