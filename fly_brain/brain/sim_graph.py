"""CUDA-Graph version of the whole-CNS simulator: same model as sim.py, static shapes only.

sim.py is event-driven with data-dependent shapes (torch.nonzero, repeat_interleave without a
size), which forces a GPU->CPU sync several times per 0.1 ms step; a 1 s run is ~10,000 steps of
Python overhead no matter how small the brain's activity is. Here every step has fixed shapes:

  * spiking neurons are compressed into a K-slot buffer with cumsum + scatter (no nonzero)
  * spike delivery expands into exactly S synapse slots per step; a filler slot pads the
    remainder, real synapses beyond S are dropped and an overflow flag is raised at the end
  * the synaptic delay is a (delay, K) ring that shifts every step instead of indexing by t
  * refractoriness is a per-neuron countdown instead of an absolute deadline

so `steps_per_graph` steps can be captured once into a torch.cuda.CUDAGraph and replayed.
"""
import math
import time

import numpy as np
import torch

from brain.sim import PARAMS


def _divisor_leq(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


@torch.no_grad()
def simulate_graph(brain, stim_idx, stim_hz, readout_idx=(), params=None, n_run=None, t_run=None,
                   bin_ms=10.0, seed=0, silence_idx=(), progress=True, stim_seg_ms=None, device="cuda",
                   k_spk=None, s_max=None, steps_per_graph=None, use_graph=True, compile=True):
    """Same contract and return dict as brain.sim.simulate (plus "device", "graph_steps").

    k_spk: spike slots per step (default 128 + 48 * n_run); s_max: synapse slots per step
    (default 65536 + 12288 * n_run). Both raise RuntimeError at the end if they overflowed.
    """
    p = dict(PARAMS, **(params or {}))
    B = n_run or p["n_run"]
    T = t_run or p["t_run"]
    N, dt = brain.n, p["dt"]
    steps = int(round(T / dt))
    delay = int(round(p["t_dly"] / dt))
    refr_steps = int(round(p["t_rfc"] / dt))
    steps_per_bin = int(round(bin_ms / dt))
    n_bins = int(math.ceil(T / bin_ms))
    dev = torch.device(device)
    if dev.type != "cuda":
        use_graph = False
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed(seed)

    K = int(k_spk or 128 + 48 * B)
    S = int(s_max or 65536 + 12288 * B)
    BN = B * N

    eg = math.exp(-dt / p["tau"])
    em = math.exp(-dt / p["t_mbr"])
    kg = p["tau"] / (p["tau"] - p["t_mbr"]) * (eg - em)
    u_th = p["v_th"] - p["v_0"]
    u_rst = p["v_rst"] - p["v_0"]
    kick = p["w_syn"] * p["f_poi"]
    w_syn = p["w_syn"]

    # connectome with a dummy neuron N (no outputs) and S filler synapses of weight 0
    col_ptr, post_d, weight_d = brain.on(dev)
    col_ptr_ext = torch.cat([col_ptr, col_ptr[-1:]])
    post_ext = torch.cat([post_d, B * N + torch.arange(S, dtype=post_d.dtype, device=dev)])  # scratch slots
    weight_ext = torch.cat([weight_d, torch.zeros(S, dtype=weight_d.dtype, device=dev)])
    filler_start = torch.tensor([post_d.numel()], dtype=torch.int64, device=dev)
    zero_i = torch.zeros(1, dtype=torch.int64, device=dev)
    S_t = torch.tensor(S, dtype=torch.int64, device=dev)

    # state
    u = torch.zeros(BN, device=dev)
    g_ext = torch.zeros(BN + S, device=dev)  # [:BN] conductances, [BN:] scratch hit by filler synapses
    g = g_ext[:BN]
    refr = torch.zeros(BN, dtype=torch.int32, device=dev)
    counts = torch.zeros(BN, dtype=torch.int32, device=dev)
    silent = torch.zeros(N, dtype=torch.bool, device=dev)
    if len(silence_idx):
        silent[torch.as_tensor(np.asarray(silence_idx, dtype=np.int64), device=dev)] = True
    silent = silent.repeat(B)

    ar_B = torch.arange(B, device=dev)
    stim_idx = torch.as_tensor(np.asarray(stim_idx, dtype=np.int64), device=dev)
    rates = np.asarray(stim_hz, dtype=np.float32)
    segmented = rates.ndim == 2
    if segmented:
        seg_rates = torch.as_tensor(rates, device=dev) * (dt / 1000.0)
        steps_per_seg = int(round(stim_seg_ms / dt))
        stim_p = seg_rates[:, 0].repeat(B).clone()
    else:
        steps_per_seg = steps_per_bin
        stim_p = torch.as_tensor(rates * (dt / 1000.0), device=dev).repeat(B).clone()
    stim_flat = (ar_B[:, None] * N + stim_idx[None, :]).reshape(-1)
    is_stim = torch.zeros(BN, dtype=torch.bool, device=dev)
    is_stim[stim_flat] = True
    kick_t = torch.tensor(kick, device=dev)
    u_rst_t = torch.tensor(u_rst, device=dev)
    refr_t = torch.tensor(refr_steps, dtype=torch.int32, device=dev)

    readout_idx = np.asarray(readout_idx, dtype=np.int64)
    R = len(readout_idx)
    readout_flat = (ar_B[:, None] * N + torch.as_tensor(readout_idx, device=dev)[None, :]).reshape(-1)
    readout = torch.zeros(B * R, n_bins, dtype=torch.int32, device=dev)
    readout_u = torch.zeros(B * R, n_bins, dtype=torch.float64, device=dev)
    readout_cur = torch.zeros(B * R, dtype=torch.int32, device=dev)
    readout_u_cur = torch.zeros(B * R, dtype=torch.float64, device=dev)
    pop = torch.zeros(n_bins, dtype=torch.int64, device=dev)
    pop_cur = torch.zeros((), dtype=torch.int64, device=dev)
    overflow = torch.zeros((), dtype=torch.bool, device=dev)

    ring = torch.zeros(delay, K, dtype=torch.int64, device=dev)   # spike indices, slot 0 newest
    ring_n = torch.zeros(delay, dtype=torch.int64, device=dev)    # valid count per slot
    mask_prev = torch.zeros(BN, dtype=torch.bool, device=dev)  # spikes of the previous step (reset is deferred)
    mask = torch.zeros(BN, dtype=torch.bool, device=dev)
    ar_K = torch.arange(K, device=dev)
    ar_1K = torch.arange(1, K + 1, dtype=torch.int32, device=dev)
    ar_S = torch.arange(S, device=dev)
    N_t = torch.tensor(N, dtype=torch.int64, device=dev)
    zero_f = torch.tensor(0.0, device=dev)

    def fused(u, g, refr, mask_prev, mask, silent, is_stim, counts):
        """One pass over all B*N neurons: apply last step's reset, integrate, threshold, count."""
        u1 = torch.where(mask_prev, u_rst_t, u)
        g1 = torch.where(mask_prev, zero_f, g)
        r1 = torch.where(mask_prev, refr_t, refr) - 1
        active = r1 <= 0
        u2 = torch.where(active, u1 * em + g1 * kg, u1)
        g2 = torch.where(active, g1 * eg, g1)
        m = (u2 > u_th) & active & ~silent
        u.copy_(u2); g.copy_(g2); refr.copy_(r1); mask.copy_(m); counts.add_(m)
        return (m & ~is_stim).sum()

    fused_fn = fused
    if compile and dev.type == "cuda":
        try:
            fused_fn = torch.compile(fused, dynamic=False, fullgraph=True)
            fused_fn(u, g, refr, mask_prev, mask, silent, is_stim, counts)  # compile now (state is all zero: harmless)
            u.zero_(); g.zero_(); refr.zero_(); mask.zero_(); counts.zero_()
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile unavailable ({type(e).__name__}: {str(e)[:80]}); running the fused step eagerly")
            fused_fn = fused

    def step():
        # 1+2. reset of last step's spikers, integration, threshold: one fused pass
        n_pop = fused_fn(u, g, refr, mask_prev, mask, silent, is_stim, counts)
        if R:
            readout_u_cur.add_(u[readout_flat].to(torch.float64))
        # compress spikers into K slots: k-th spike sits where the running count first reaches k
        cnt = torch.cumsum(mask, 0, dtype=torch.int32)
        n = cnt[-1]
        spk_pos = torch.searchsorted(cnt, ar_1K)  # K int64 positions; >= n are invalid
        overflow.logical_or_(n > K)
        # 3. deliver the spikes emitted `delay` steps ago (oldest ring slot)
        arr, arr_n = ring[delay - 1], ring_n[delay - 1]
        valid = ar_K < arr_n
        trial = torch.where(valid, arr // N, zero_i)
        pre = torch.where(valid, arr % N, N_t)
        starts = col_ptr_ext[pre]
        lens = col_ptr_ext[pre + 1] - starts
        excl = torch.cumsum(lens, 0) - lens
        lens_c = torch.clamp(torch.minimum(lens, S_t - excl), min=0)
        used = lens_c.sum()
        overflow.logical_or_(used < lens.sum())
        lens_all = torch.cat([lens_c, (S_t - used).reshape(1)])
        starts_all = torch.cat([starts, filler_start])
        trial_all = torch.cat([trial, zero_i])
        excl_all = torch.cumsum(lens_all, 0) - lens_all
        pos = torch.repeat_interleave(starts_all - excl_all, lens_all, output_size=S) + ar_S
        tr = torch.repeat_interleave(trial_all, lens_all, output_size=S)
        g_ext.index_add_(0, tr * N + post_ext[pos], weight_ext[pos] * w_syn)
        # 4. Poisson stimulus (default CUDA generator: graph-safe)
        fire = torch.rand(stim_flat.numel(), device=dev) < stim_p
        u.index_add_(0, stim_flat, fire.to(u.dtype) * kick_t)
        # 5. bookkeeping (the reset of this step's spikers happens at the start of the next step)
        pop_cur.add_(n_pop)
        if R:
            readout_cur.add_(mask[readout_flat])
        mask_prev.copy_(mask)
        # 6. push this step's spikes into the ring
        ring.copy_(torch.roll(ring, 1, 0))
        ring_n.copy_(torch.roll(ring_n, 1, 0))
        ring[0].copy_(spk_pos)
        ring_n[0].copy_(n)

    def reset_state():
        for x in (u, g_ext, refr, counts, readout_cur, readout_u_cur, pop_cur, overflow, ring, ring_n, mask, mask_prev):
            x.zero_()

    G = steps_per_graph or _divisor_leq(math.gcd(steps_per_bin, steps_per_seg), 100)
    graph = None
    if use_graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(G):
                step()
        reset_state()
        torch.cuda.manual_seed(seed)  # note: the capture consumed random numbers; replays draw fresh ones

    t0 = time.time()
    t, seg = 0, -1
    next_report = steps // 10
    while t < steps:
        if segmented and t // steps_per_seg != seg:
            seg = t // steps_per_seg
            stim_p.copy_(seg_rates[:, min(seg, seg_rates.shape[1] - 1)].repeat(B))
        if graph is not None and t + G <= steps and (t + G - 1) // steps_per_bin == t // steps_per_bin:
            graph.replay()
            t += G
        else:
            step()
            t += 1
        if t % steps_per_bin == 0 or t == steps:
            b = (t - 1) // steps_per_bin
            readout[:, b].copy_(readout_cur); readout_cur.zero_()
            readout_u[:, b].copy_(readout_u_cur); readout_u_cur.zero_()
            pop[b].copy_(pop_cur); pop_cur.zero_()
        if progress and t >= next_report:
            print(f"  {100 * t // steps:3d}%  {time.time() - t0:6.1f}s", flush=True)
            next_report += steps // 10
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    seconds = time.time() - t0
    if bool(overflow.item()):
        raise RuntimeError(f"static buffers overflowed (k_spk={K}, s_max={S}); raise them and rerun")
    rate = counts.view(B, N).float().mean(0) / (T / 1000.0)
    return {"rate": rate.cpu().numpy(), "readout": readout.view(B, R, n_bins).cpu().numpy(), "bin_ms": bin_ms,
            "pop_hz": pop.cpu().numpy() / (B * bin_ms / 1000.0),
            "readout_v": (readout_u / steps_per_bin + p["v_0"]).view(B, R, n_bins).float().cpu().numpy(),
            "n_run": B, "t_run": T, "seconds": seconds, "device": str(dev), "graph_steps": G if graph else 0,
            "k_spk": K, "s_max": S}
