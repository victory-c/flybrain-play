"""Passenger screen: a PD rider steers the Tarmac while the whole brain watches through its senses.

For every neuron we accumulate its rate against the bike's lean, lean rate, steer angle and lateral
position over all riders and 10 ms chunks, then rank neurons and cell types by correlation. This
says which descending and motor neurons the connectome makes *carry* the bike's lean, before any
decoder is fitted. Also reports where the population activity sits (per superclass).

usage: python -m runs.screen --riders 8 --seconds 6 --out results/screen.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bike.dynamics import Peloton
from bike.readout import Readout
from bike.senses import Senses
from bike.tarmac import tarmac_sl9
from brain.loop import BrainLoop
from brain.sim import W_SYN_MALE_CNS, Brain

ROOT = Path(__file__).resolve().parents[1]
VARS = ["phi", "phi_dot", "delta", "y", "psi_dot", "v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--riders", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warmup-ms", type=float, default=300.0)
    ap.add_argument("--ctrl-ms", type=float, default=10.0)
    ap.add_argument("--v0", type=float, default=4.0)
    ap.add_argument("--gust", type=float, default=25.0)
    ap.add_argument("--phi0", type=float, default=4.0)
    ap.add_argument("--goal", default="none")
    ap.add_argument("--w-syn-scale", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-ms", type=float, default=40.0, help="readout smoothing, as in runs/ride.py")
    ap.add_argument("--out", default="results/screen.json")
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    meta = pd.read_parquet(ROOT / "brain_meta.parquet")
    senses = Senses(meta, goal=a.goal, device=dev)
    readout = Readout(meta, device=dev)
    ro_idx = torch.as_tensor(readout.idx, device=dev)
    brain = Brain(ROOT / "brain.npz")
    w = W_SYN_MALE_CNS if a.w_syn_scale is None else 0.275 * a.w_syn_scale
    loop = BrainLoop(brain, a.riders, senses.idx, [], params={"w_syn": w}, device=dev, seed=a.seed)
    bikes = Peloton(a.riders, tarmac_sl9(), device=dev, seed=a.seed)
    bikes.reset(a.v0, a.phi0, a.gust, 1.0)
    N, B, dt = brain.n, a.riders, a.ctrl_ms / 1000.0
    n_sub = int(round(a.ctrl_ms / (bikes.dt * 1000.0)))
    print(f"screen: {B} riders, {a.seconds:.0f} s, PD rider steering, gusts {a.gust} Nm, {len(senses.idx)} driven neurons, on {dev}", flush=True)

    loop.set_rates(senses.rates(bikes.state, bikes.psi_dot))
    loop.run(a.warmup_ms)
    warm = loop.take_counts().float().sum(0) / (a.warmup_ms / 1000.0) / B  # Hz per neuron at cruise
    V = len(VARS)
    S_r = torch.zeros(N, device=dev, dtype=torch.float64)
    S_rr = torch.zeros(N, device=dev, dtype=torch.float64)
    S_rx = torch.zeros(V, N, device=dev, dtype=torch.float64)
    S_x = torch.zeros(V, device=dev, dtype=torch.float64)
    S_xx = torch.zeros(V, device=dev, dtype=torch.float64)
    n = 0
    smooth = torch.zeros(B, readout.n_features, device=dev)
    samples = {"features": [], "torque": [], "state": [], "alive": []}
    t0 = time.time()
    steps = int(round(a.seconds / dt))
    for k in range(steps):
        loop.run(a.ctrl_ms)
        counts = loop.take_counts()
        r = counts.double() / dt  # (B, N) Hz in this chunk
        smooth += (readout.features(counts[:, ro_idx], a.ctrl_ms) - smooth) * min(1.0, a.ctrl_ms / a.tau_ms)
        s = bikes.state
        torque = bikes.pd_oracle()
        samples["features"].append(smooth.cpu().numpy().astype(np.float32))
        samples["torque"].append(torque.cpu().numpy().astype(np.float32))
        samples["state"].append(s.cpu().numpy().astype(np.float32))
        samples["alive"].append((~bikes.done).cpu().numpy())
        x = torch.stack([s[:, 4], s[:, 6], s[:, 5], s[:, 1], bikes.psi_dot, s[:, 3]], 0).double()  # (V, B)
        alive = (~bikes.done).double()
        S_r += (r * alive[:, None]).sum(0)
        S_rr += (r * r * alive[:, None]).sum(0)
        S_rx += (x * alive) @ r
        S_x += (x * alive).sum(1)
        S_xx += (x * x * alive).sum(1)
        n += int(alive.sum())
        bikes.step(torque, torch.full((B,), 60.0, device=dev), torch.zeros(B, device=dev), n_sub=n_sub)
        loop.set_rates(senses.rates(bikes.state, bikes.psi_dot))
        if (k + 1) % int(round(1.0 / dt)) == 0:
            print(f"  t={(k + 1) * dt:4.1f}s  alive {int((~bikes.done).sum())}/{B}  mean|phi| {np.degrees(s[:, 4].abs().mean().item()):4.1f} deg"
                  f"  pop {float(r.sum(1).mean()):,.0f} sp/s  wall {time.time() - t0:.0f}s", flush=True)

    mean_r = (S_r / n).cpu().numpy()
    var_r = (S_rr / n).cpu().numpy() - mean_r ** 2
    mx = (S_x / n).cpu().numpy()
    vx = (S_xx / n).cpu().numpy() - mx ** 2
    cov = (S_rx / n).cpu().numpy() - np.outer(mx, mean_r)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(np.outer(vx, var_r))
    corr = np.nan_to_num(corr)

    df = meta[["idx", "bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide"]].copy()
    df["cruise_hz"] = warm.cpu().numpy()
    df["mean_hz"] = mean_r
    df["driven"] = False
    df.loc[senses.idx, "driven"] = True
    for i, vn in enumerate(VARS):
        df[f"r_{vn}"] = corr[i]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)

    print("\n== population by superclass (Hz mean per neuron, and total spikes/s per brain)")
    g = df[~df.driven].groupby("superclass").agg(n=("idx", "size"), hz=("mean_hz", "mean"), total=("mean_hz", "sum")).sort_values("total", ascending=False)
    print(g.round(1).head(15).to_string())
    print(f"\ndriven neurons total {df[df.driven].mean_hz.sum():,.0f} sp/s; rest of brain {df[~df.driven].mean_hz.sum():,.0f} sp/s; "
          f"{int((df.mean_hz > 1).sum()):,} neurons > 1 Hz, {int((df.mean_hz > 20).sum()):,} > 20 Hz")

    active = df[(~df.driven) & (df.mean_hz > 0.5)]
    print(f"\n== descending neurons above 0.5 Hz ({int((active.superclass == 'descending_neuron').sum())} of 1314)")
    dn = active[active.superclass == "descending_neuron"].sort_values("mean_hz", ascending=False)
    print(dn[["type", "instance", "somaSide", "cruise_hz", "mean_hz", "r_phi", "r_phi_dot", "r_delta", "r_y"]].head(40).round(2).to_string(index=False))
    print(f"\n== motor neurons above 0.5 Hz ({int(active.superclass.isin(['vnc_motor', 'cb_motor']).sum())} of 815)")
    mn = active[active.superclass.isin(["vnc_motor", "cb_motor"])].sort_values("mean_hz", ascending=False)
    print(mn[["type", "instance", "somaSide", "superclass", "cruise_hz", "mean_hz", "r_phi", "r_phi_dot", "r_delta"]].head(40).round(2).to_string(index=False))
    for vn in ("phi", "phi_dot", "delta"):
        out = active[active.superclass.isin(["descending_neuron", "vnc_motor", "cb_motor", "ascending_neuron", "sensory_ascending"])]
        out = out.reindex(out[f"r_{vn}"].abs().sort_values(ascending=False).index)
        print(f"\n== output-side neurons most correlated with {vn}")
        print(out[["type", "instance", "somaSide", "superclass", "mean_hz", f"r_{vn}", "r_phi", "r_phi_dot", "r_delta"]].head(25).round(2).to_string(index=False))
    # type-level L/R asymmetry: does (L - R) of a type track lean?
    rows = []
    for (ty, sc), grp in active[active.superclass.isin(["descending_neuron", "vnc_motor", "cb_motor"])].groupby(["type", "superclass"]):
        L, R = grp[grp.somaSide == "L"], grp[grp.somaSide == "R"]
        if len(L) and len(R):
            rows.append({"type": ty, "superclass": sc, "n": len(grp), "hz": grp.mean_hz.mean(),
                         "rphi_L": L.r_phi.mean(), "rphi_R": R.r_phi.mean(), "rdot_L": L.r_phi_dot.mean(), "rdot_R": R.r_phi_dot.mean()})
    if rows:
        tl = pd.DataFrame(rows)
        tl["asym"] = (tl.rphi_L - tl.rphi_R).abs() + (tl.rdot_L - tl.rdot_R).abs()
        print("\n== output types whose left and right members disagree about lean (candidates for steering)")
        print(tl.sort_values("asym", ascending=False).head(30).round(2).to_string(index=False))

    (ROOT / a.out).write_text(json.dumps({"config": vars(a), "n_samples": n, "vars": VARS,
                                          "superclass": g.reset_index().to_dict(orient="records"),
                                          "neurons": df[(df.mean_hz > 0.5) | df.driven].round(4).to_dict(orient="records")}))
    df.to_parquet(ROOT / a.out.replace(".json", ".parquet"))
    np.savez_compressed(ROOT / a.out.replace(".json", "_samples.npz"), feature_names=np.array(readout.names),
                        **{k: np.stack(v) for k, v in samples.items()})
    print(f"-> {a.out.replace('.json', '_samples.npz')}: {len(samples['torque'])} chunks x {B} riders of readout features + teacher torque")
    print(f"\n-> {a.out} (+ .parquet with every neuron)   brain {loop.seconds:.0f}s wall")


if __name__ == "__main__":
    main()
