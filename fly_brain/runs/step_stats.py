import sys; sys.path.insert(0, '/home/s/st/stevejobs/flybrain/fly_brain')
import numpy as np, pandas as pd
from pathlib import Path
from brain.sim import W_SYN_MALE_CNS, Brain, simulate
from brain.taste import DRINKS, COCKTAILS, neuron_input, taste_matrix, taste_vector
ROOT = Path('/home/s/st/stevejobs/flybrain/fly_brain')
brain = Brain(ROOT/'brain.npz'); meta = pd.read_parquet(ROOT/'brain_meta.parquet')
grn_idx, M, organ = taste_matrix(meta)
for name, x in [('orange', DRINKS["🍊 Χυμός πορτοκάλι"]), ('cola', DRINKS["🥤 Κόλα"]), ('negroni', COCKTAILS["🥃 Negroni"])]:
    t = taste_vector(x); u = neuron_input(t, M); on = u >= 1.0
    for B in (1, 8):
        st = {}
        r = simulate(brain, grn_idx[on], u[on], n_run=B, t_run=1000.0, params={"w_syn": W_SYN_MALE_CNS}, progress=False, device='cuda', stats=st)
        print(f"{name:8s} B={B}  max spikes/step {st['max_spk']:6d}  max syn/step {st['max_syn']:9,d}  mean syn/step {st['sum_syn']/10000:9,.0f}  {r['seconds']:.1f}s", flush=True)
