# Slurm job 36088: inline Python sent to srun on stdin, recovered from the Claude session transcript.
# timeout 900 srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet ../venv-cuda/bin/python -
# (run from fly_brain/)
"""Which sensory drive saturates the central brain? Each population alone for 1 s at its cruise rate."""
import numpy as np, pandas as pd, torch
from bike.senses import Senses
from bike.dynamics import Peloton
from brain.loop import BrainLoop
from brain.sim import W_SYN_MALE_CNS, Brain
dev = "cuda"
meta = pd.read_parquet("brain_meta.parquet")
senses = Senses(meta, goal="odor", device=dev)
brain = Brain("brain.npz")
bikes = Peloton(1, device=dev); bikes.reset(4.0, 0.0)
bikes.state[0, 6] = 0.3  # a little roll rate so the roll channels are on
P = senses.population_rates(bikes.state, torch.tensor([0.1], device=dev))[0]
print("cruise population rates:", {n: round(float(v), 1) for n, v in zip(senses.names, P)})
loop = BrainLoop(brain, 1, senses.idx, [], params={"w_syn": W_SYN_MALE_CNS}, device=dev)
sc = meta["superclass"].astype(str).to_numpy(); ty = meta["type"].astype(str).to_numpy()
driven = np.zeros(brain.n, bool); driven[senses.idx] = True
def cond(label, keep):
    loop.reset()
    mask = torch.tensor([n in keep for n in senses.names], device=dev, dtype=torch.float32)
    loop.set_rates((P * mask) @ senses.E)
    tot = []
    for k in range(5):
        _, pop = loop.run(200.0); tot.append(float(pop[0]) / 0.2)
    c = loop.take_counts()[0].cpu().numpy() / 1.0
    c[driven] = 0
    hot = c > 100
    top = pd.Series(c).groupby(ty).mean().sort_values(ascending=False).head(6)
    cb = c[sc == "cb_intrinsic"].sum(); vnc = c[sc == "vnc_intrinsic"].sum(); dn = c[sc == "descending_neuron"].sum()
    print(f"{label:28s} pop/200ms: {' '.join(f'{t:>8,.0f}' for t in tot)}  | >100Hz {hot.sum():5d} | cb {cb:,.0f} vnc {vnc:,.0f} DN {dn:,.0f} sp/s | top: "
          + ", ".join(f"{t} {v:.0f}" for t, v in top.items()), flush=True)
for n in ["VS", "HS", "HALT", "JO", "LEG1", "ORN"]:
    cond(n + " only", {n + "_L", n + "_R"})
cond("all but ORN", {n for n in senses.names if not n.startswith("ORN")})
cond("all but ORN, JO", {n for n in senses.names if not n.startswith(("ORN", "JO"))})
cond("all", set(senses.names))
