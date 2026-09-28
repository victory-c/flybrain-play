"""Profile the static-shape step (eager, no graph) to see which kernels dominate."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.profiler import profile, ProfilerActivity
from brain.sim import W_SYN_MALE_CNS, Brain
from brain.sim_graph import simulate_graph
from brain.taste import DRINKS, neuron_input, taste_matrix, taste_vector
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=8); a = ap.parse_args()
brain = Brain(ROOT / "brain.npz"); meta = pd.read_parquet(ROOT / "brain_meta.parquet")
grn_idx, M, organ = taste_matrix(meta)
t = taste_vector(DRINKS["🍊 Χυμός πορτοκάλι"]); u = neuron_input(t, M); on = u >= 1.0
kw = dict(readout_idx=[306, 6367], n_run=a.trials, t_run=20.0, bin_ms=10.0, params={"w_syn": W_SYN_MALE_CNS}, progress=False, device="cuda", use_graph=False)
simulate_graph(brain, grn_idx[on], u[on], **kw)  # warm-up
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    r = simulate_graph(brain, grn_idx[on], u[on], **kw)
print(f"B={a.trials}: 200 steps eager {r['seconds']*1000:.0f} ms -> {r['seconds']*1000/200:.3f} ms/step")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=60))
