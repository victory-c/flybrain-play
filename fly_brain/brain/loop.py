"""The whole-CNS simulator with its state kept between calls, so a body can sit in the loop.

Same neuron model, constants and update order as brain/sim.py's simulate(); the differences are
  * the state (v, g, refractory clocks, the spike delay queue) persists across run() calls,
  * every trial in the batch gets its own stimulus rates (set_rates takes a (n_run, n_stim) array),
  * run(ms) returns the spike counts of the readout neurons and of the whole population in that window.

    loop = BrainLoop(brain, n_run=32, stim_idx=sensory, readout_idx=descending, device="cuda")
    loop.set_rates(hz)            # (32, n_sensory) Hz
    counts, pop = loop.run(10.0)  # 10 ms of brain time -> (32, n_descending) spikes, (32,) spikes
"""
import math
import time

import numpy as np
import torch

from .sim import PARAMS


class BrainLoop:
    def __init__(self, brain, n_run, stim_idx, readout_idx, params=None, device=None, seed=0, silence_idx=()):
        self.p = p = dict(PARAMS, **(params or {}))
        self.brain, self.B, self.N, self.dt = brain, n_run, brain.n, p["dt"]
        self.dev = dev = torch.device(device or "cpu")
        self.delay = int(round(p["t_dly"] / self.dt))
        self.refr_steps = int(round(p["t_rfc"] / self.dt))
        self.col_ptr, self.post, self.weight = brain.on(dev)
        eg, em = math.exp(-self.dt / p["tau"]), math.exp(-self.dt / p["t_mbr"])
        self.eg, self.em = eg, em
        self.kg = p["tau"] / (p["tau"] - p["t_mbr"]) * (eg - em)
        self.u_th, self.u_rst = p["v_th"] - p["v_0"], p["v_rst"] - p["v_0"]
        self.kick = torch.tensor(p["w_syn"] * p["f_poi"], device=dev)
        self.w_syn = p["w_syn"]
        B, N = self.B, self.N
        ar_B = torch.arange(B, device=dev)

        self.stim_idx = torch.as_tensor(np.asarray(stim_idx, dtype=np.int64), device=dev)
        self.n_stim = len(self.stim_idx)
        self.stim_flat = (ar_B[:, None] * N + self.stim_idx[None, :]).reshape(-1)
        self.stim_p = torch.zeros(B * self.n_stim, device=dev)
        self.is_stim = torch.zeros(B * N, dtype=torch.bool, device=dev)
        self.is_stim[self.stim_flat] = True

        readout_idx = np.asarray(readout_idx, dtype=np.int64)
        self.R = len(readout_idx)
        self.readout_pos = torch.full((N,), -1, dtype=torch.int64, device=dev)
        self.readout_pos[torch.as_tensor(readout_idx, device=dev)] = torch.arange(self.R, device=dev)

        self.silent = torch.zeros(N, dtype=torch.bool, device=dev)
        if len(silence_idx):
            self.silent[torch.as_tensor(np.asarray(silence_idx, dtype=np.int64), device=dev)] = True
        self.silent = self.silent.repeat(B)
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        dev, B, N = self.dev, self.B, self.N
        self.gen = torch.Generator(device=dev).manual_seed(self.seed if seed is None else seed)
        self.u = torch.zeros(B * N, device=dev)
        self.g = torch.zeros(B * N, device=dev)
        self.refr_until = torch.zeros(B * N, dtype=torch.int32, device=dev)
        self.counts_all = torch.zeros(B * N, dtype=torch.int32, device=dev)  # every neuron, since take_counts()
        self.queue = [torch.empty(0, dtype=torch.int64, device=dev)] * self.delay
        self.t = 0
        self.seconds = 0.0

    def set_rates(self, rates_hz):
        """rates_hz: (n_run, n_stim) Hz, or (n_stim,) for the same rates in every trial."""
        r = torch.as_tensor(np.asarray(rates_hz, dtype=np.float32) if not torch.is_tensor(rates_hz) else rates_hz,
                            dtype=torch.float32, device=self.dev)
        if r.dim() == 1:
            r = r[None, :].expand(self.B, -1)
        self.stim_p = (r * (self.dt / 1000.0)).reshape(-1)

    def take_counts(self):
        """Spike counts of every neuron since the last call: (B, N) int32; resets the accumulator."""
        c = self.counts_all.view(self.B, self.N).clone()
        self.counts_all.zero_()
        return c

    @torch.no_grad()
    def run(self, ms):
        """Advance the brain by `ms`. Returns (readout spike counts (B, R) int32, population spikes (B,) int64)."""
        B, N, R, dev = self.B, self.N, self.R, self.dev
        steps = int(round(ms / self.dt))
        counts = torch.zeros(B * R, dtype=torch.int32, device=dev)
        pop = torch.zeros(B, dtype=torch.int64, device=dev)
        u, g, refr_until, queue = self.u, self.g, self.refr_until, self.queue
        col_ptr, post_d, weight_d = self.col_ptr, self.post, self.weight
        rep = torch.repeat_interleave
        t0 = time.time()
        for _ in range(steps):
            t = self.t
            active = refr_until <= t
            torch.where(active, u * self.em + g * self.kg, u, out=u)
            torch.where(active, g * self.eg, g, out=g)
            spk = torch.nonzero((u > self.u_th) & active & ~self.silent).squeeze(1)

            arriving = queue[t % self.delay]
            if arriving.numel():
                trial, pre = arriving // N, arriving % N
                starts = col_ptr[pre]
                lens = col_ptr[pre + 1] - starts
                total = int(lens.sum())
                if total:
                    pos = rep(starts - (torch.cumsum(lens, 0) - lens), lens) + torch.arange(total, device=dev)
                    g.index_add_(0, rep(trial, lens) * N + post_d[pos], weight_d[pos] * self.w_syn)

            fire = torch.rand(self.stim_flat.numel(), generator=self.gen, device=dev) < self.stim_p
            u.index_add_(0, self.stim_flat, fire.to(u.dtype) * self.kick)

            if spk.numel():
                u[spk] = self.u_rst
                g[spk] = 0.0
                refr_until[spk] = t + self.refr_steps
                self.counts_all[spk] += 1
                unstim = spk[~self.is_stim[spk]]
                pop += torch.bincount(unstim // N, minlength=B)
                rp = self.readout_pos[spk % N]
                keep = rp >= 0
                if keep.any():
                    flat = (spk[keep] // N) * R + rp[keep]
                    counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
            queue[t % self.delay] = spk
            self.t = t + 1
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self.seconds += time.time() - t0
        return counts.view(B, R), pop
