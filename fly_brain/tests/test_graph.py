"""Checks of the CUDA-Graph simulator: toy networks (same as test_sim) + whole brain vs sim.py.

usage: python -m tests.test_graph [--device cuda]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain.sim import W_SYN_MALE_CNS, Brain, simulate
from brain.sim_graph import simulate_graph
from brain.taste import DRINKS, neuron_input, taste_matrix, taste_vector
from tests.test_sim import toy_brain

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-brain", action="store_true")
    a = ap.parse_args()
    d = a.device
    b = toy_brain([(0, 1, 100.0), (0, 2, -100.0)], 3)
    r = simulate_graph(b, [0], [100.0], n_run=50, t_run=1000.0, progress=False, device=d)
    print(f"graph: driven neuron at 100 Hz input -> {r['rate'][0]:.1f} Hz, partner {r['rate'][1]:.1f} Hz, "
          f"inhibited {r['rate'][2]:.1f} Hz  (graph steps {r['graph_steps']})")
    assert 90 < r["rate"][0] <= 101 and r["rate"][1] > 0 and r["rate"][2] == 0
    alone = simulate_graph(b, [2], [50.0], n_run=50, t_run=1000.0, progress=False, device=d)["rate"][2]
    inhib = simulate_graph(b, [0, 2], [150.0, 50.0], n_run=50, t_run=1000.0, progress=False, device=d)["rate"][2]
    print(f"graph: Poisson-driven neuron 2 alone {alone:.1f} Hz, with inhibition {inhib:.1f} Hz")
    assert 40 < alone < 60 and 40 < inhib < 60
    silent = simulate_graph(b, [0], [100.0], n_run=10, t_run=500.0, silence_idx=[1], progress=False, device=d)["rate"]
    assert silent[1] == 0
    # readout + segmented stimulus: rate 100 Hz for 500 ms then 0
    r = simulate_graph(b, [0], [[100.0, 0.0]], readout_idx=[0, 1], n_run=20, t_run=1000.0, bin_ms=100.0,
                       stim_seg_ms=500.0, progress=False, device=d)
    hz = r["readout"].mean(0) / 0.1
    print("graph: segmented stimulus, neuron 0 per 100 ms bin:", hz[0].round(0).tolist())
    assert hz[0][:5].min() > 80 and hz[0][6:].max() < 5
    print("toy checks passed")
    if a.skip_brain:
        return

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    by_instance = meta.set_index("instance")["idx"]
    mn9 = np.array([by_instance["MN9_L"], by_instance["MN9_R"]])
    grn_idx, M, organ = taste_matrix(meta)
    t = taste_vector(DRINKS["🍊 Χυμός πορτοκάλι"]); u = neuron_input(t, M); on = u >= 1.0
    kw = dict(readout_idx=mn9, n_run=8, t_run=1000.0, bin_ms=50.0, params={"w_syn": W_SYN_MALE_CNS}, progress=False, device=d)
    ref = simulate(brain, grn_idx[on], u[on], **kw)
    new = simulate_graph(brain, grn_idx[on], u[on], **kw)
    for name, r in (("sim.py", ref), ("graph", new)):
        mn = r["readout"][:, 0].sum(1).mean()
        print(f"{name:7s} MN9_L {mn:5.1f} Hz  spikes/s {r['rate'].sum():,.0f}  neurons>10Hz {(r['rate']>10).sum():,}  "
              f"pop_hz[-1] {r['pop_hz'][-1]:,.0f}  {r['seconds']:.1f}s")
    assert abs(new["rate"].sum() - ref["rate"].sum()) / ref["rate"].sum() < 0.1
    assert abs(new["readout"][:, 0].sum(1).mean() - ref["readout"][:, 0].sum(1).mean()) < 15
    print("whole-brain agreement OK")


if __name__ == "__main__":
    main()
