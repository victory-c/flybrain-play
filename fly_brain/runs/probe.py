"""Stimulus-response probe: which output neurons does a sensory population drive, left vs right?

Closed-loop correlations mix everything that moves together (in the passenger screen, heading swings in
antiphase with roll rate, so every roll-rate neuron looks like a heading neuron). This probe removes the
confound: the bike is held upright at cruise speed, every sensory population fires at its cruise rate, and
one population gets extra drive on the LEFT in half the brains and on the RIGHT in the other half. The
per-neuron difference (left-probe minus right-probe) is the population's lateralised effect.

usage: python -m runs.probe --pop HS --extra 30 --brains 16 --ms 1000
       python -m runs.probe --pop HS VS HALT JO LEG1 --out results/probe.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bike.dynamics import Peloton
from bike.senses import Senses
from brain.loop import BrainLoop
from brain.sim import W_SYN_MALE_CNS, Brain

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", nargs="+", default=["HS", "VS"])
    ap.add_argument("--extra", type=float, default=30.0, help="extra Hz on one side")
    ap.add_argument("--brains", type=int, default=16, help="half get the left probe, half the right")
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--warmup-ms", type=float, default=300.0)
    ap.add_argument("--v", type=float, default=4.5)
    ap.add_argument("--lane", default="hs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/probe.json")
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    senses = Senses(meta, device=dev, lane=a.lane)
    brain = Brain(ROOT / "brain.npz")
    B = a.brains - a.brains % 2
    bikes = Peloton(B, device=dev)
    bikes.reset(a.v, 0.0)
    base = senses.population_rates(bikes.state, bikes.psi_dot)  # (B, P) cruise rates, upright
    out_mask = meta.superclass.isin(["descending_neuron", "vnc_motor", "cb_motor"]).to_numpy()
    report = {}
    for pop in a.pop:
        P = base.clone()
        iL, iR = senses.names.index(f"{pop}_L"), senses.names.index(f"{pop}_R")
        P[: B // 2, iL] += a.extra
        P[B // 2:, iR] += a.extra
        loop = BrainLoop(brain, B, senses.idx, [], params={"w_syn": W_SYN_MALE_CNS}, device=dev, seed=1)
        loop.set_rates(P @ senses.E)
        loop.run(a.warmup_ms)
        loop.take_counts()
        loop.run(a.ms)
        hz = loop.take_counts().float().cpu().numpy() / (a.ms / 1000.0)  # (B, N)
        L, R = hz[: B // 2], hz[B // 2:]
        diff = L.mean(0) - R.mean(0)
        se = np.sqrt(L.var(0) / len(L) + R.var(0) / len(R)) + 1e-6
        df = meta[["type", "instance", "somaSide", "superclass"]].copy()
        df["hz_L"], df["hz_R"], df["diff"], df["z"] = L.mean(0), R.mean(0), diff, diff / se
        o = df[out_mask & ((df.hz_L > 1) | (df.hz_R > 1))].reindex(df[out_mask].z.abs().sort_values(ascending=False).index).dropna()
        print(f"\n== {pop}: +{a.extra:.0f} Hz on the left vs on the right ({B // 2} brains each, {a.ms:.0f} ms)")
        print(o[["type", "instance", "somaSide", "superclass", "hz_L", "hz_R", "diff", "z"]].head(20).round(2).to_string(index=False))
        report[pop] = o.head(60).round(3).to_dict(orient="records")
    (ROOT / a.out).write_text(json.dumps(report, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()
