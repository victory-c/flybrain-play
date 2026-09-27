"""CPU vs GPU benchmark of the whole-CNS simulator: orange juice, 1 s, growing trial batches.

usage: python -m runs.bench_gpu --devices cpu cuda --trials 1 8 32 128
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brain.sim import W_SYN_MALE_CNS, Brain, simulate
from brain.taste import DRINKS, neuron_input, taste_matrix, taste_vector

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--trials", nargs="+", type=int, default=[1, 8, 32, 128])
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--out", default="results/bench_gpu.json")
    args = ap.parse_args()
    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    by_instance = meta.set_index("instance")["idx"]
    mn9 = np.array([by_instance["MN9_L"], by_instance["MN9_R"]])
    grn_idx, M, organ = taste_matrix(meta)
    t = taste_vector(DRINKS["🍊 Χυμός πορτοκάλι"])
    u = neuron_input(t, M)
    on = u >= 1.0
    print(f"torch {torch.__version__}  cpu threads {torch.get_num_threads()}  "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    rows = []
    for dev in args.devices:
        if dev == "cuda" and not torch.cuda.is_available():
            print("cuda not available, skipping"); continue
        simulate(brain, grn_idx[on], u[on], readout_idx=mn9, n_run=1, t_run=100.0, bin_ms=50.0,
                 params={"w_syn": W_SYN_MALE_CNS}, progress=False, device=dev)  # warm-up
        for n in args.trials:
            if dev == "cpu" and n > 32:
                continue
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            r = simulate(brain, grn_idx[on], u[on], readout_idx=mn9, n_run=n, t_run=args.ms, bin_ms=50.0,
                         params={"w_syn": W_SYN_MALE_CNS}, progress=False, device=dev)
            mn9_hz = r["readout"][:, 0].sum(1) / (args.ms / 1000.0)
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
            row = {"device": dev, "trials": n, "seconds": round(r["seconds"], 2),
                   "s_per_trial_s": round(r["seconds"] / n / (args.ms / 1000.0), 3),
                   "mn9_hz": round(float(mn9_hz.mean()), 1), "spikes_per_s": round(float(r["rate"].sum())),
                   "gpu_gb": round(mem, 2)}
            rows.append(row)
            print(f"{dev:5s} trials {n:4d}  {row['seconds']:8.1f} s  {row['s_per_trial_s']:7.2f} s per simulated second  "
                  f"MN9 {row['mn9_hz']:5.1f} Hz  gpu {row['gpu_gb']} GB", flush=True)
            (ROOT / args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
