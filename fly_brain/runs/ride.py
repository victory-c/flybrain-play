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
  python -m runs.ride --replay results/ride.json --brain-out results/ride_brain.npz   # ... and record the whole brain
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
from bike.readout import READOUTS, Decoder, Readout
from bike.senses import Senses
from bike.tarmac import eigen_speeds, tarmac_sl9, whipple_matrices
from brain.loop import BrainLoop
from brain.sim import W_SYN_MALE_CNS, Brain

ROOT = Path(__file__).resolve().parents[1]


class BrainRecorder:
    """Every rider's whole brain during a replay, for the brain map on the 3D page. Same point cloud and regions as
    the drinks dashboard (export/export_dashboard.py): per bin, the rate of each of the 140,638 neurons with a 3D
    position as uint8 (255 = CLOUD_HZ), and each region's mean rate with the driven sensory neurons left out."""

    def __init__(self, meta, stim_idx, bin_ms, device):
        from export.export_dashboard import CLOUD_HZ, REGIONS, region_of
        meta = meta.sort_values("idx")
        reg = np.array([region_of(t, c, s) for t, c, s in zip(meta["type"], meta["class"], meta["superclass"])])
        self.reg_ids = [r for r, _ in REGIONS]
        self.reg_labels = [lab for _, lab in REGIONS]
        point_idx = np.fromfile(ROOT / "web_data" / "brain_idx.bin", dtype=np.int32).astype(np.int64)
        self.point_region = np.array([self.reg_ids.index(r) for r in reg[point_idx]], dtype=np.uint8)
        self.point_idx = torch.as_tensor(point_idx, device=device)
        driven = np.zeros(len(reg), bool)
        driven[np.asarray(stim_idx)] = True
        M = np.zeros((len(reg), len(self.reg_ids)), np.float32)
        for j, r in enumerate(self.reg_ids):
            m = (reg == r) & ~driven
            if m.any():
                M[m, j] = 1.0 / m.sum()
        self.M = torch.as_tensor(M, device=device)
        self.bin_ms, self.cloud_hz = bin_ms, CLOUD_HZ
        self.act, self.reg = [], []

    def add(self, counts):
        """counts (B, N): every neuron's spikes in one bin."""
        hz = counts.float() / (self.bin_ms / 1000.0)
        self.act.append((hz[:, self.point_idx] * (255.0 / self.cloud_hz)).clamp(0, 255).round().to(torch.uint8).cpu())
        self.reg.append((hz @ self.M).cpu())

    def save(self, path):
        act = torch.stack(self.act).numpy()  # (bins, B, points)
        np.savez_compressed(path, act=act, region_hz=torch.stack(self.reg).numpy().round(2), bin_ms=self.bin_ms,
                            cloud_hz=self.cloud_hz, regions=np.array(self.reg_ids), labels=np.array(self.reg_labels),
                            point_region=self.point_region)
        print(f"brain -> {path}  ({act.shape[0]} bins of {self.bin_ms:.0f} ms, {act.shape[1]} riders, {act.shape[2]:,} points)")


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
    ap.add_argument("--init-theta", default=None, help="results json whose best_theta seeds CEM (e.g. a DAgger result)")
    ap.add_argument("--lane-cap", type=float, default=25.0, help="cap on the squared lateral offset in the lane penalty (m^2)")
    ap.add_argument("--offroad", type=float, default=0.0, help="if > 0: leaving the road (|y| > this, m) ends a rider's run like a fall")
    ap.add_argument("--gains", default="", help="sensory gain overrides for bike/senses.py GAINS, e.g. 'hs_heading=200,hs_lane=60'")
    ap.add_argument("--polarity", default="legacy", choices=["legacy", "physio"], help="which eye's VS/HS cells a rotation drives (bike/senses.py)")
    ap.add_argument("--readout", default="base", choices=["base", "lane"], help="readout neuron set (bike/readout.py READOUTS)")
    ap.add_argument("--sigma-new", type=float, default=0.8, help="CEM width for decoder weights absent from --init-theta")
    ap.add_argument("--lane-filter", default=None, help="runs/probe.py json: add one matched-filter lane signal (HS) with one gain")
    ap.add_argument("--sigma-lane-filter", type=float, default=2.0, help="CEM width for the lane-filter gain")
    ap.add_argument("--center-all", action="store_true", help="read every steering feature as its deviation from the warm-up rate")
    ap.add_argument("--tau-lane-ms", type=float, default=40.0, help="smoothing of the lane channel (LANE_DNS + lane filter); roll stays at --tau-ms")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--sigma0", type=float, default=0.15, help="CEM search width around the teacher fit")
    ap.add_argument("--lane-penalty", type=float, default=0.01, help="fitness cost per m^2 of lateral offset per second")
    ap.add_argument("--dagger", type=int, default=0, help="DAgger rounds instead of CEM: label visited states with the PD rider, refit")
    ap.add_argument("--dagger-beta0", type=float, default=0.5, help="round-1 probability of executing the PD rider's action; halves each round")
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
    ap.add_argument("--brain-out", default=None, help="replay/open loop: record every rider's whole brain to this npz (brain map)")
    ap.add_argument("--brain-bin-ms", type=float, default=100.0, help="time bin of the brain recording")
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
            meta = self.meta = pd.read_parquet(ROOT / "brain_meta.parquet")
            gains = {kv.split("=")[0].strip(): float(kv.split("=")[1]) for kv in a.gains.split(",") if "=" in kv}
            self.senses = Senses(meta, goal=a.goal, device=self.dev, lane=a.lane, gains=gains, polarity=a.polarity)
            if gains:
                print("sensory gain overrides:", gains, flush=True)
            self.readout = Readout(meta, types=READOUTS[a.readout], device=self.dev)
            lf = lane_filter_from_probe(a.lane_filter, self.readout) if a.lane_filter else None
            self.decoder = Decoder(self.readout, lane_filter=lf, center_all=a.center_all)
            print("senses :", self.senses.describe())
            print("readout:", self.readout.describe(), flush=True)
            brain = Brain(ROOT / "brain.npz")
            w = W_SYN_MALE_CNS if a.w_syn_scale is None else 0.275 * a.w_syn_scale
            self.loop = BrainLoop(brain, a.riders, self.senses.idx, self.readout.idx,
                                  params={"w_syn": w}, device=self.dev, seed=a.seed)
            print(f"brain: {brain.n:,} neurons, {len(self.senses.idx)} driven, {len(self.readout.idx)} read out, "
                  f"w_syn {w:.4f} mV", flush=True)

    @torch.no_grad()
    def episode(self, theta=None, oracle=False, seed=0, log=None, verbose=False, collect=None, beta=0.0, brain=None):
        """collect: dict with lists 'X', 'y' -> appends (readout features / rate_scale, PD rider's torque) for every
        alive rider and step (DAgger labels). beta: probability per rider and step of executing the PD rider's torque.
        brain: a BrainRecorder, fed every neuron's spikes per bin from the end of warm-up on."""
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
            slow = smooth.clone()
            self.decoder.set_center(smooth)
            if brain is not None:
                loop.take_counts()  # drop the warm-up
                per_bin = max(1, int(round(brain.bin_ms / a.ctrl_ms)))
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
                if brain is not None and (k + 1) % per_bin == 0:
                    brain.add(loop.take_counts())
                f = self.readout.features(counts, a.ctrl_ms)
                smooth += (f - smooth) * min(1.0, a.ctrl_ms / a.tau_ms)
                slow += (f - slow) * min(1.0, a.ctrl_ms / a.tau_lane_ms)
                gf_hist.append(self.readout.gf_spikes(counts))
                gf_hist = gf_hist[-max(1, int(round(50.0 / a.ctrl_ms))):]
                if not a.no_bail:
                    bail = (torch.stack(gf_hist).sum(0) >= 4) & alive
                    bikes.done |= bail
                    bailed |= bail
                steer, power, brake = self.decoder.act(theta, smooth, slow if a.tau_lane_ms != a.tau_ms else None)
                if collect is not None or beta > 0:
                    expert = bikes.pd_oracle()
                    if collect is not None and alive.any():
                        collect["X"].append((smooth[alive] / self.decoder.rate_scale).cpu().numpy())
                        collect["y"].append(expert[alive].cpu().numpy())
                    if beta > 0:
                        use = torch.rand(B, device=dev) < beta
                        steer = torch.where(use, expert.clamp(-self.decoder.steer_max, self.decoder.steer_max), steer)
                pop_sum += float(pop.float().mean()) / dt
                n_chunks += 1
            bikes.step(steer, power, brake, n_sub=n_sub)
            s = bikes.state
            if a.offroad > 0:
                bikes.done |= s[:, 1].abs() > a.offroad
            al = alive.float()
            upright += al * dt
            fitness += al * dt * (1.0 + 0.02 * s[:, 3] - a.lane_penalty * (s[:, 1] ** 2).clamp(max=a.lane_cap) - 0.002 * steer ** 2)
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
            src = json.loads((ROOT / a.replay).read_text())
            best = src["best_theta"]
            if src.get("decoder_names") and src["decoder_names"] != dec_names(dec):
                old = dict(zip(src["decoder_names"], best))
                best = [old.get(n, 0.0) for n in dec_names(dec)]
                print("replay: decoder weights mapped by name onto this readout", flush=True)
            theta = torch.tensor(best, device=ride.dev)[None, :].repeat(a.riders, 1)
            label = f"replay of {a.replay}"
        else:
            theta = torch.tensor(dec.prior_mean(), device=ride.dev)[None, :].repeat(a.riders, 1)
            label = "open loop (prior decoder: no steering, ~70 W)"
        log = []
        rec = BrainRecorder(ride.meta, ride.senses.idx, a.brain_bin_ms, ride.dev) if a.brain_out else None
        r = ride.episode(theta, seed=a.seed, log=log, verbose=True, brain=rec)
        print(f"{label}:", ride.summary(r))
        if rec is not None:
            rec.save(ROOT / a.brain_out)
        trace = {"mode": label, "state_names": STATE, "dn_names": ride.readout.names, "sense_names": ride.senses.names,
                 "theta": theta[0].cpu().tolist(), "theta_names": dec_names(dec), "trace": log,
                 "result": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}}
        (ROOT / a.trace).write_text(json.dumps(trace))
        print("trace ->", a.trace)
        return

    if a.dagger:
        return dagger(a, ride, dec)

    # cross-entropy method over decoder weights
    rng = np.random.default_rng(a.seed)
    mu, sigma = dec.prior_mean(), dec.prior_sigma()
    if a.init:
        w, b, r2 = fit_teacher(ROOT / a.init, dec, a.ridge, ride.readout.names)
        mu[:dec.F], mu[dec.F] = w, b
        sigma[:dec.F + 1] = a.sigma0
        print(f"teacher fit: R^2 {r2:.2f} on the screen's samples; CEM starts from it", flush=True)
    if a.init_theta:
        src = json.loads((ROOT / a.init_theta).read_text())
        old = dict(zip(src.get("decoder_names", dec_names(dec)), src["best_theta"]))
        names = dec_names(dec)
        mu = np.array([old.get(n, 0.0) for n in names], dtype=np.float32)
        sigma[:dec.F + 1] = a.sigma0
        new = [i for i, n in enumerate(names) if n not in old]
        sigma[new] = a.sigma_new
        print(f"CEM starts from best_theta of {a.init_theta}; {len(new)} new decoder weights start at 0 with width {a.sigma_new}", flush=True)
    if dec.lane_filter is not None:
        sigma[-1] = a.sigma_lane_filter
        print(f"lane-filter gain starts at {mu[-1]:.2f} with width {a.sigma_lane_filter}", flush=True)
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


def ridge_fit(X, T, dec, ridge):
    """atanh(torque / steer_max) ~ X w + b; returns (w, b, R^2)."""
    y = np.arctanh(np.clip(T / dec.steer_max, -0.95, 0.95))
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], 1)
    reg = ridge * np.eye(X.shape[1] + 1); reg[-1, -1] = 0.0
    wb = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    r2 = 1.0 - ((Xb @ wb - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return wb[:-1].astype(np.float32), np.float32(wb[-1]), float(r2)


def dagger(a, ride, dec):
    """DAgger (Ross et al. 2011): the fly rides with the current decoder (mixed with the PD rider with probability
    beta), every visited state is labelled with the PD rider's torque, all data so far is refit by ridge regression.
    The PD rider keeps the lane (it leans toward the lane centre), so the lane-keeping part of steering is learned
    wherever the readout neurons carry lane information."""
    z = np.load(ROOT / a.init, allow_pickle=True)
    assert list(z["feature_names"]) == list(ride.readout.names), "screen and ride use different readout populations"
    al = z["alive"].reshape(-1).astype(bool)
    Xs = [z["features"].reshape(-1, dec.F)[al] / dec.rate_scale]
    Ts = [z["torque"].reshape(-1)[al]]
    theta_np = dec.prior_mean()
    history, best, t_all = [], None, time.time()
    for it in range(a.dagger + 1):
        w, b, r2 = ridge_fit(np.concatenate(Xs).astype(np.float64), np.concatenate(Ts).astype(np.float64), dec, a.ridge)
        theta_np[:dec.F], theta_np[dec.F] = w, b
        theta = torch.tensor(theta_np, device=ride.dev)[None, :].repeat(a.riders, 1)
        beta = a.dagger_beta0 * 0.5 ** it if it < a.dagger - 1 else 0.0   # last rounds: the fly alone
        col = {"X": [], "y": []}
        r = ride.episode(theta, seed=a.seed + 1000 * (it + 1), verbose=(it == 0), collect=col, beta=beta)
        n_new = sum(len(x) for x in col["X"])
        if n_new:
            Xs.append(np.concatenate(col["X"])); Ts.append(np.concatenate(col["y"]))
        row = {"round": it, "beta": beta, "fit_r2": r2, "samples": int(sum(len(x) for x in Xs)), "fitness_mean": float(r["fitness"].mean()),
               "upright_mean": float(r["upright"].mean()), "finish_frac": float((r["upright"] >= a.seconds - 1e-6).mean()),
               "lateral_abs_mean": float(np.abs(r["lateral"]).mean()), "distance_mean": float(r["distance"].mean()), "wall_s": r["wall"]}
        history.append(row)
        print(f"dagger {it:2d}  beta {beta:5.3f}  fit R^2 {r2:.2f}  samples {row['samples']:,}  fitness {row['fitness_mean']:6.2f}  " + ride.summary(r), flush=True)
        if beta == 0.0 and (best is None or row["fitness_mean"] > best[0]):
            best = (row["fitness_mean"], theta_np.copy(), it)
        (ROOT / a.out).write_text(json.dumps({
            "config": vars(a), "method": "dagger", "device": ride.dev, "decoder_names": dec_names(dec), "dn_names": ride.readout.names,
            "sense_names": ride.senses.names, "history": history, "best_round": best[2] if best else None,
            "best_theta": (best[1] if best else theta_np).tolist(), "best_fitness": best[0] if best else None,
            "wall_seconds": time.time() - t_all}, indent=1))
    print(f"done: best pure-policy round {best[2] if best else '-'}, results -> {a.out}  ({(time.time() - t_all) / 60:.1f} min)")


def dec_names(dec):
    names = [f"steer:{n}" for n in dec.r.names] + ["steer:bias", "pedal:gain", "pedal:bias", "brake:gain", "brake:bias"]
    return names + (["steer:lane_filter"] if dec.lane_filter is not None else [])


def lane_filter_from_probe(path, readout, pop="HS", zmin=3.0):
    """Matched filter over readout features from a runs/probe.py result: each (type, side) feature gets its mean
    left-minus-right response (Hz) to the probe; normalised so the probe's own response pattern reads as 1.0."""
    recs = json.loads((ROOT / path).read_text())[pop]
    by = {}
    for r in recs:
        if abs(r["z"]) >= zmin:
            by.setdefault((r["type"], r["somaSide"]), []).append(r["diff"])
    v = np.array([np.mean(by[(t, s)]) if (t, s) in by else 0.0 for t, s, _, _ in readout.groups], dtype=np.float32)
    used = [f"{t}_{s} {np.mean(by[(t, s)]):+.1f}" for t, s, _, _ in readout.groups if (t, s) in by]
    print(f"lane filter from {path}: {len(used)} features: " + ", ".join(used), flush=True)
    return v / max(float((v ** 2).sum()), 1e-6)


if __name__ == "__main__":
    main()
