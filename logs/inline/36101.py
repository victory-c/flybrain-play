# Slurm job 36101: inline Python sent to srun on stdin, recovered from the Claude session transcript.
# timeout 900 srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -
# (run from fly_brain/)
"""Is a weak odour goal stable? ORN_DM1 at several rates on top of the cruising senses, 1.5 s each."""
import numpy as np, pandas as pd, torch
from bike.senses import Senses
from bike.dynamics import Peloton
from brain.loop import BrainLoop
from brain.sim import W_SYN_MALE_CNS, Brain
dev = "cuda"
meta = pd.read_parquet("brain_meta.parquet")
senses = Senses(meta, goal="odor", device=dev)
brain = Brain("brain.npz")
bikes = Peloton(1, device=dev); bikes.reset(4.0, 0.0); bikes.state[0, 6] = 0.3
P = senses.population_rates(bikes.state, torch.tensor([0.1], device=dev))[0]
loop = BrainLoop(brain, 1, senses.idx, [], params={"w_syn": W_SYN_MALE_CNS}, device=dev)
sc = meta["superclass"].astype(str).to_numpy(); ty = meta["type"].astype(str).to_numpy()
orn = torch.tensor([n.startswith("ORN") for n in senses.names], device=dev)
for hz in [0, 2, 4, 8, 12]:
    loop.reset(); p = P.clone(); p[orn] = hz; loop.set_rates(p @ senses.E)
    tot = []
    for k in range(6):
        _, pop = loop.run(250.0); tot.append(float(pop[0]) / 0.25)
    c = loop.take_counts()[0].cpu().numpy() / 1.5
    apl = c[ty == "APL"].mean(); kc = c[meta["class"].astype(str).to_numpy() == "Kenyon_Cell"].mean()
    print(f"ORN {hz:2d} Hz  pop per 250ms: {' '.join(f'{t:>8,.0f}' for t in tot)}  | APL {apl:5.0f} Hz  KC {kc:5.1f} Hz  cb {c[sc=='cb_intrinsic'].sum():,.0f} sp/s", flush=True)
