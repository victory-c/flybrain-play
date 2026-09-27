"""Serve one custom drink to the fly and report whether it extends its proboscis (MN9_L).

usage: python -m runs.serve "🧋 奶茶" --sugar 80 --caffeine 150 --ph 6.5
       python -m runs.serve "🍶 清酒" --abv 15 --sugar 5 --ph 4.3 --hunger 0.5 --trials 10

Ingredients (same slots as brain/taste.py DRINKS): sugar g/L, abv %, caffeine mg/L, ibu (hops /
bitters), salt g/L, co2 g/L, ph. --hunger 0..1 scales the bitter channel by (1 - hunger), as in
runs/hunger.py, so a hungry fly tolerates bitter drinks.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain.sim import W_SYN_MALE_CNS, Brain, simulate
from brain.taste import CHANNELS, neuron_input, taste_matrix, taste_vector

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--sugar", type=float, default=0.0, help="g/L")
    ap.add_argument("--abv", type=float, default=0.0, help="%% alcohol")
    ap.add_argument("--caffeine", type=float, default=0.0, help="mg/L")
    ap.add_argument("--ibu", type=float, default=0.0, help="hops / bitters")
    ap.add_argument("--salt", type=float, default=0.0, help="g/L")
    ap.add_argument("--co2", type=float, default=0.0, help="g/L")
    ap.add_argument("--ph", type=float, default=7.0)
    ap.add_argument("--hunger", type=float, default=0.0, help="0 fed .. 1 starving")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--ms", type=float, default=1000.0)
    a = ap.parse_args()

    x = np.array([a.sugar, a.abv, a.caffeine, a.ibu, a.salt, a.co2, a.ph], dtype=float)
    t = taste_vector(x)
    t[CHANNELS.index("bitter")] *= 1.0 - a.hunger

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    by_instance = meta.set_index("instance")["idx"]
    mn9 = np.array([by_instance["MN9_L"], by_instance["MN9_R"]])
    grn_idx, M, organ = taste_matrix(meta)
    u = neuron_input(t, M)
    on = u >= 1.0

    print(f"{a.name}  taste = [" + " ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, t)) + "]"
          f"   hunger {a.hunger:.2f}   {int(on.sum())} taste neurons driven", flush=True)
    r = simulate(brain, grn_idx[on], u[on], readout_idx=mn9, n_run=a.trials, t_run=a.ms, bin_ms=50.0,
                 params={"w_syn": W_SYN_MALE_CNS}, progress=False)
    per_trial = r["readout"][:, 0].sum(1) / (a.ms / 1000.0)
    hz = per_trial.mean()
    active = int((r["rate"] > 10).sum())
    verdict = "🪰👅 伸出口器，喜欢！" if hz >= 10 else ("🪰🤔 犹豫" if hz > 0 else "🪰🚫 拒绝")
    print(f"MN9_L {hz:6.1f} Hz  ({', '.join(f'{v:.0f}' for v in per_trial)})   "
          f"{active:,} neurons >10 Hz   {r['seconds']:.0f}s   {verdict}")


if __name__ == "__main__":
    main()
