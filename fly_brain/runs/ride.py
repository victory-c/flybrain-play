"""Teach the fly to ride a Specialized S-Works Tarmac SL9.

Closed loop: bike state -> sensory neurons (bike/senses.py) -> whole male CNS (brain/loop.py) ->
descending neurons -> decoder (bike/readout.py) -> steer torque, pedal power, brake -> bike
(bike/dynamics.py). The connectome is fixed; the decoder's weights are learned by a cross-entropy
method over a peloton of riders that all run in one batch on the GPU.

usage:
  python -m runs.ride --oracle                         # a PD rider, no brain: is the bike rideable?
  python -m runs.ride --open-loop --riders 8           # brain in the loop, decoder at its prior
  python -m runs.ride --riders 32 --generations 12     # learn the decoder (results/ride.json)
  python -m runs.ride --replay results/ride.json       # ride the best decoder again, log every step
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bike.dynamics import STATE, Peloton
from bike.readout import Decoder, Readout
from bike.senses import Senses
from bike.tarmac import eigen_speeds, tarmac_sl9, whipple_matrices
from brain.loop import BrainLoop
from brain.sim import W_SYN_MALE_CNS, Brain

ROOT = Path(__file__).resolve().parents[1]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--riders", type=int, default=32, help="parallel bikes = brain batch size")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=10.0, help="episode length, bike time")
    ap.add_argument("--warmup-ms", type=float, default=300.0, help="brain settles on the upright bike first")
    ap.add_argument("--ctrl-ms", type=float, default=10.0, help="brain chunk = control period")
    ap.add_argument("--v0", type=float, default=4.0, help="initial speed m/s (Tarmac + 70 kg is self-stable 4.7..7.7)")
    ap.add_argument("--rider-mass", type=float, default=70.0)
    ap.add_argument("--phi0", type=float, default=2.0, help="initial lean sd, deg")
    ap.add_argument("--gust", type=float, default=15.0, help="side-gust roll torque sd, Nm")
    ap.add_argument("--gust-tau", type=float, default=1.0)
    ap.add_argument("--lane", default="hs", choices=["hs", "none"], help="lane cue as HS optic-flow asymmetry")
    ap.add_argument("--goal", default="none", choices=["odor", "none"], help="odor goal saturates the antennal lobe at this synapse scale (see bike/README.md)")
    ap.add_argument("--no-bail", action="store_true", help="ignore giant-fibre escapes")
    ap.add_argument("--tau-ms", type=float, default=40.0, help="readout smoothing time constant")
    ap.add_argument("--init", default=None, help="screen *_samples.npz: fit the steering decoder to the teacher first")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--sigma0", type=float, default=0.15, help="CEM search width around the teacher fit")
    ap.add_argument("--lane-penalty", type=float, default=0.01, help="fitness cost per m^2 of lateral offset per second")
    ap.add_argument("--elite", type=float, default=0.25)
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--w-syn-scale", type=float, default=None, help="override synapse scale (default 0.45)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ride.json")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--open-loop", action="store_true")
    ap.add_argument("--replay", default=None, help="results json with best_theta")
    ap.add_argument("--trace", default="results/ride_trace.json")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


class Ride:
    def __init__(self, a):
        self.a = a
        self.dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
        params = tarmac_sl9(a.rider_mass)
        self.bikes = Peloton(a.riders, params, device=self.dev, seed=a.seed)
        wv, cv = eigen_speeds(whipple_matrices(params))
        print(f"Tarmac SL9 + {a.rider_mass:.0f} kg: {self.bikes.mT:.1f} kg, self-stable {wv:.2f}..{cv:.2f} m/s; "
              f"start at {a.v0:.1f} m/s, gusts {a.gust:.0f} Nm, {a.riders} riders on {self.dev}", flush=True)
        self.stable = (wv, cv)
        self.loop = self.senses = self.readout = self.decoder = None
        if not a.oracle:
            meta = pd.read_parquet(ROOT / "brain_meta.parquet")
            self.senses = Senses(meta, goal=a.goal, device=self.dev, lane=a.lane)
            self.readout = Readout(meta, device=self.dev)
            self.decoder = Decoder(self.readout)
            print("senses :", self.senses.describe())
            print("readout:", self.readout.describe(), flush=True)
            brain = Brain(ROOT / "brain.npz")
            w = W_SYN_MALE_CNS if a.w_syn_scale is None else 0.275 * a.w_syn_scale
            self.loop = BrainLoop(brain, a.riders, self.senses.idx, self.readout.idx,
                                  params={"w_syn": w}, device=self.dev, seed=a.seed)
            print(f"brain: {brain.n:,} neurons, {len(self.senses.idx)} driven, {len(self.readout.idx)} read out, "
                  f"w_syn {w:.4f} mV", flush=True)

    @torch.no_grad()
    def episode(self, theta=None, oracle=False, seed=0, log=None, verbose=False):
        a, bikes, loop = self.a, self.bikes, self.loop
        B, dev = bikes.n, bikes.dev
        dt = a.ctrl_ms / 1000.0
        n_sub = int(round(a.ctrl_ms / (bikes.dt * 1000.0)))
        bikes.reset(a.v0, a.phi0, a.gust, a.gust_tau)
        t_wall = time.time()
        if not oracle:
            F = self.readout.n_features
            loop.reset(seed)
            loop.set_rates(self.senses.rates(bikes.state, bikes.psi_dot))
            counts, pop = loop.run(a.warmup_ms)
            smooth = self.readout.features(counts, a.warmup_ms)
            if verbose:
                f0 = smooth.mean(0).cpu().numpy()
                print("  after warm-up, DN rates (Hz): " + ", ".join(f"{n} {v:.0f}" for n, v in zip(self.readout.names, f0)
                                                                    if v >= 0.5) or "  all DNs silent")
                print(f"  population {float(pop.float().mean()) / (a.warmup_ms / 1000):,.0f} spikes/s per brain", flush=True)
            gf_hist = []
        fitness = torch.zeros(B, device=dev)
        upright = torch.zeros(B, device=dev)
        bailed = torch.zeros(B, dtype=torch.bool, device=dev)
        pop_sum, n_chunks = 0.0, 0
        steps = int(round(a.seconds / dt))
        for k in range(steps):
            alive = ~bikes.done
            if oracle:
                steer = bikes.pd_oracle()
                power = torch.full((B,), 60.0, device=dev)
                brake = torch.zeros(B, device=dev)
            else:
                counts, pop = loop.run(a.ctrl_ms)
                f = self.readout.features(counts, a.ctrl_ms)
                smooth += (f - smooth) * min(1.0, a.ctrl_ms / a.tau_ms)
                gf_hist.append(self.readout.gf_spikes(counts))
                gf_hist = gf_hist[-max(1, int(round(50.0 / a.ctrl_ms))):]
                if not a.no_bail:
                    bail = (torch.stack(gf_hist).sum(0) >= 4) & alive
                    bikes.done |= bail
                    bailed |= bail
                steer, power, brake = self.decoder.act(theta, smooth)
                pop_sum += float(pop.float().mean()) / dt
                n_chunks += 1
            bikes.step(steer, power, brake, n_sub=n_sub)
            s = bikes.state
            al = alive.float()
            upright += al * dt
            fitness += al * dt * (1.0 + 0.02 * s[:, 3] - a.lane_penalty * (s[:, 1] ** 2).clamp(max=25.0) - 0.002 * steer ** 2)
            if not oracle:
                loop.set_rates(self.senses.rates(s, bikes.psi_dot))
            if log is not None:
                row = {"t": round((k + 1) * dt, 3), "state": s.cpu().numpy().round(4).tolist(),
                       "steer": steer.cpu().numpy().round(3).tolist(), "power": power.cpu().numpy().round(1).tolist(),
                       "brake": brake.cpu().numpy().round(3).tolist(), "done": bikes.done.cpu().tolist()}
                if not oracle:
                    row["dn_hz"] = smooth.cpu().numpy().round(1).tolist()
                    row["sense_hz"] = self.senses.population_rates(s, bikes.psi_dot).cpu().numpy().round(1).tolist()
                    row["pop_hz"] = (pop.float() / dt).cpu().numpy().round(0).tolist()
                log.append(row)
            if bikes.done.all():
                break
        fitness -= 2.0 * bailed.float()
        s = bikes.state
        return {"fitness": fitness.cpu().numpy(), "upright": upright.cpu().numpy(), "distance": s[:, 0].cpu().numpy(),
                "lateral": s[:, 1].cpu().numpy(), "speed": s[:, 3].cpu().numpy(), "bailed": bailed.cpu().numpy(),
                "pop_hz": pop_sum / max(1, n_chunks), "wall": time.time() - t_wall}

    def summary(self, r):
        return (f"upright {r['upright'].mean():5.2f} s (max {r['upright'].max():5.2f}, {100 * (r['upright'] >= self.a.seconds - 1e-6).mean():3.0f}% finish)"
                f"  dist {r['distance'].mean():5.1f} m  |y| {np.abs(r['lateral']).mean():4.2f} m  v {r['speed'].mean():4.2f}"
                f"  bailed {int(r['bailed'].sum())}  pop {r['pop_hz']:,.0f} sp/s  {r['wall']:.0f}s")


def main():
    a = parse()
    ride = Ride(a)
    out = ROOT / a.out
    out.parent.mkdir(exist_ok=True)

    if a.oracle:
        r = ride.episode(oracle=True, seed=a.seed, log=(log := []))
        print("PD oracle:", ride.summary(r))
        (ROOT / a.trace).write_text(json.dumps({"mode": "oracle", "state_names": STATE, "trace": log}))
        return

    dec = ride.decoder
    D = dec.D
    if a.replay or a.open_loop:
        if a.replay:
            best = json.loads((ROOT / a.replay).read_text())["best_theta"]
            theta = torch.tensor(best, device=ride.dev)[None, :].repeat(a.riders, 1)
            label = f"replay of {a.replay}"
        else:
            theta = torch.tensor(dec.prior_mean(), device=ride.dev)[None, :].repeat(a.riders, 1)
            label = "open loop (prior decoder: no steering, ~70 W)"
        log = []
        r = ride.episode(theta, seed=a.seed, log=log, verbose=True)
        print(f"{label}:", ride.summary(r))
        trace = {"mode": label, "state_names": STATE, "dn_names": ride.readout.names, "sense_names": ride.senses.names,
                 "theta": theta[0].cpu().tolist(), "theta_names": dec_names(dec), "trace": log,
                 "result": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}}
        (ROOT / a.trace).write_text(json.dumps(trace))
        print("trace ->", a.trace)
        return

    # cross-entropy method over decoder weights
    rng = np.random.default_rng(a.seed)
    mu, sigma = dec.prior_mean(), dec.prior_sigma()
    if a.init:
        w, b, r2 = fit_teacher(ROOT / a.init, dec, a.ridge, ride.readout.names)
        mu[:dec.F], mu[dec.F] = w, b
        sigma[:dec.F + 1] = a.sigma0
        print(f"teacher fit: R^2 {r2:.2f} on the screen's samples; CEM starts from it", flush=True)
    n_elite = max(2, int(round(a.elite * a.riders)))
    history, best_theta, best_fit = [], mu.copy(), -np.inf
    t_all = time.time()
    for gen in range(a.generations):
        theta_np = mu[None, :] + sigma[None, :] * rng.standard_normal((a.riders, D)).astype(np.float32)
        theta_np[0] = mu
        theta = torch.tensor(theta_np, device=ride.dev)
        r = ride.episode(theta, seed=a.seed + 1000 * (gen + 1), verbose=(gen == 0))
        fit = r["fitness"]
        order = np.argsort(-fit)
        elite = theta_np[order[:n_elite]]
        mu = elite.mean(0)
        sigma = np.maximum(elite.std(0), a.sigma_min).astype(np.float32)
        if fit[order[0]] > best_fit:
            best_fit, best_theta = float(fit[order[0]]), theta_np[order[0]].copy()
        row = {"gen": gen, "best": float(fit.max()), "mean": float(fit.mean()), "elite_mean": float(fit[order[:n_elite]].mean()),
               "upright_mean": float(r["upright"].mean()), "upright_best_rider": float(r["upright"][order[0]]),
               "finish_frac": float((r["upright"] >= a.seconds - 1e-6).mean()), "distance_mean": float(r["distance"].mean()),
               "bailed": int(r["bailed"].sum()), "pop_hz": r["pop_hz"], "wall_s": r["wall"], "sigma_mean": float(sigma.mean())}
        history.append(row)
        print(f"gen {gen:2d}  fit best {row['best']:6.2f} mean {row['mean']:6.2f}  " + ride.summary(r), flush=True)
        (ROOT / a.out).write_text(json.dumps({
            "config": vars(a), "device": ride.dev, "stable_speeds": ride.stable, "decoder_names": dec_names(dec),
            "dn_names": ride.readout.names, "sense_names": ride.senses.names, "history": history,
            "best_fitness": best_fit, "best_theta": best_theta.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(),
            "brain_seconds": ride.loop.seconds, "wall_seconds": time.time() - t_all}, indent=1))
    print(f"done: best fitness {best_fit:.2f}, results -> {a.out}  ({(time.time() - t_all) / 60:.1f} min)")


def fit_teacher(path, dec, ridge, names):
    """Ridge regression of atanh(teacher torque / steer_max) on the smoothed readout features."""
    z = np.load(path, allow_pickle=True)
    assert list(z["feature_names"]) == list(names), "screen and ride use different readout populations"
    alive = z["alive"].reshape(-1).astype(bool)
    X = z["features"].reshape(-1, dec.F)[alive] / dec.rate_scale
    y = np.arctanh(np.clip(z["torque"].reshape(-1)[alive] / dec.steer_max, -0.95, 0.95))
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], 1)
    reg = ridge * np.eye(dec.F + 1); reg[-1, -1] = 0.0
    wb = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    r2 = 1.0 - ((Xb @ wb - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    top = np.argsort(-np.abs(wb[:-1]))[:8]
    print("  largest steering weights: " + ", ".join(f"{names[i]} {wb[i]:+.2f}" for i in top))
    return wb[:-1].astype(np.float32), np.float32(wb[-1]), float(r2)


def dec_names(dec):
    return [f"steer:{n}" for n in dec.r.names] + ["steer:bias", "pedal:gain", "pedal:bias", "brake:gain", "brake:bias"]


if __name__ == "__main__":
    main()
