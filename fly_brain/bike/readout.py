"""From descending neurons to the handlebars.

Readout populations are identified descending neurons (DNs) of the male CNS whose roles in walking
are published (roles from Cande 2018, Bidaye 2014/2020, Rayshubskiy 2020, Namiki 2018, Sapkal 2024):

  steering       DNa01, DNa02, DNa03, DNb01 (left / right)
  forward drive  DNg100 (BDN2), DNp09 (P9), DNg97 (oDN1), DNa05, DNa07, DNp26
  backing        MDN                          -> brake
  escape         DNp01 (giant fibre)          -> the fly bails (jumps off) when it fires 4 spikes in 50 ms

The decoder maps the smoothed rates of these populations (per type, per side) to steer torque, pedal
power and braking. Its weights are the only thing that is *learned* (bike/README.md): the connectome
is never changed. Steering is a linear readout squashed by tanh; pedalling and braking are sigmoids of
the forward and backward drives.
"""
import numpy as np
import torch

# Found by runs/screen.py (a PD rider steers, the brain watches): left and right members of these types
# fire with opposite sign against the bike's roll rate (|r| up to 0.76 for DNp20) or its lean angle.
ROLL_RATE_DNS = ["DNp20", "DNp22", "DNg46", "DNge043", "DNb06", "DNp17", "DNpe013", "DNge097", "DNg94", "DNg90", "DNge088"]
LEAN_DNS = ["DNp33", "DNge091", "DNg29", "DNg41", "DNp18", "DNp73", "DNb04"]
WING_STEERING_MNS = ["b1 MN", "b2 MN", "b3 MN", "hg1 MN", "hi2 MN", "i2 MN"]  # the fly's own flight-steering output
WALKING_DNS = ["DNa01", "DNa02", "DNa03", "DNb01", "DNp09", "DNg100", "DNg97", "MDN", "DNp01"]  # literature roles
DN_TYPES = ROLL_RATE_DNS + LEAN_DNS + WING_STEERING_MNS + WALKING_DNS
# Found by runs/probe.py (bike held upright, extra drive to the left vs the right HS cells): the lateralised targets
# of the horizontal-system (yaw / translational optic flow, i.e. the lane cue), distinct from the VS-driven roll
# channel above. DNp15 is the HS-to-neck descending neuron; GNG283 is a brain motor neuron (head/neck).
LANE_DNS = ["DNa16", "DNb03", "DNa06", "DNp15", "DNge107", "DNge086", "DNge031", "DNge033", "GNG283"]
READOUTS = {"base": DN_TYPES, "lane": DN_TYPES + LANE_DNS}
FORWARD = ["DNg100", "DNp09", "DNg97"]


class Readout:
    def __init__(self, meta, types=DN_TYPES, device="cpu"):
        self.dev = torch.device(device)
        ty, side = meta["type"].astype(str), meta["somaSide"].astype(str)
        self.types = [t for t in types if (ty == t).any()]
        groups, idx, offset = [], [], 0
        for t in self.types:
            for s in ("L", "R"):
                ids = meta.loc[(ty == t) & (side == s), "idx"].to_numpy(dtype=np.int64)
                groups.append((t, s, offset, len(ids)))  # (type, side, offset into self.idx, size)
                idx.append(ids)
                offset += len(ids)
        self.idx = np.concatenate(idx)
        self.groups = groups
        self.names = [f"{t}_{s}" for t, s, _, _ in groups]
        F = len(groups)
        W = torch.zeros(len(self.idx), F, device=self.dev)
        for j, (_, _, o, n) in enumerate(groups):
            if n:
                W[o:o + n, j] = 1.0 / n
        self.W = W
        self.gf = torch.tensor([i for i, (t, _, _, _) in enumerate(groups) if t == "DNp01"], device=self.dev)
        self.forward = torch.tensor([i for i, (t, _, _, _) in enumerate(groups) if t in FORWARD], device=self.dev)
        self.mdn = torch.tensor([i for i, (t, _, _, _) in enumerate(groups) if t == "MDN"], device=self.dev)

    @property
    def n_features(self):
        return len(self.groups)

    def describe(self):
        return ", ".join(f"{t}_{s} {n}" for t, s, _, n in self.groups)

    def features(self, counts, ms):
        """spike counts (B, R) in a window of `ms` -> mean rate per (type, side) in Hz (B, F)."""
        return (counts.float() / (ms / 1000.0)) @ self.W

    def gf_spikes(self, counts):
        """giant-fibre spikes in the window, both sides (B,)."""
        cols = []
        for t, _, o, n in self.groups:
            if t == "DNp01":
                cols.extend(range(o, o + n))
        if not cols:
            return torch.zeros(counts.shape[0], dtype=counts.dtype, device=counts.device)
        return counts[:, cols].sum(1)


class Decoder:
    """theta = [steer weights (F), steer bias, pedal gain, pedal bias, brake gain, brake bias]."""

    def __init__(self, readout, steer_max=6.0, power_max=250.0, brake_max=6.0, rate_scale=50.0, lane_filter=None):
        """lane_filter: optional (F,) matched filter from runs/probe.py (each feature's response to left-vs-right HS
        drive). Its output, sum_j w_j (f_j - warm-up f_j), is one lane signal with one extra gain: theta[-1]."""
        self.r = readout
        self.F = readout.n_features
        self.D = self.F + 1 + 2 + 2
        self.lane_filter = None if lane_filter is None else torch.as_tensor(lane_filter, dtype=torch.float32, device=readout.dev)
        if self.lane_filter is not None:
            self.D += 1
        self.f0 = None
        self.steer_max, self.power_max, self.brake_max, self.rate_scale = steer_max, power_max, brake_max, rate_scale
        # lane-channel features are read as deviations from each rider's warm-up rate (they fire at ~30 Hz baseline;
        # uncentred, any weight on them is a standing steering bias). The other features stay absolute.
        self.center_mask = torch.tensor([g[0] in LANE_DNS for g in readout.groups], dtype=torch.float32, device=readout.dev)
        self.center = None

    def set_center(self, f):
        """f (B, F) smoothed Hz at the end of warm-up."""
        self.center = f * self.center_mask
        self.f0 = f.clone()

    def lane_signal(self, f):
        """(B,) matched-filter output; ~ +1 for a 60 Hz left-minus-right HS asymmetry (probe units)."""
        if self.lane_filter is None or self.f0 is None:
            return torch.zeros(f.shape[0], device=f.device)
        return (f - self.f0) @ self.lane_filter

    def prior_mean(self):
        mu = np.zeros(self.D, dtype=np.float32)
        mu[self.F + 1] = 1.0   # pedal gain
        mu[self.F + 2] = -1.0  # pedal bias: ~70 W with silent forward DNs
        mu[self.F + 3] = 1.0   # brake gain
        mu[self.F + 4] = -6.0  # brake bias: essentially off until MDN fires
        return mu

    def prior_sigma(self):
        s = np.ones(self.D, dtype=np.float32)
        s[self.F + 1:] = 0.5
        return s

    @torch.no_grad()
    def act(self, theta, f):
        """theta (B, D), f (B, F) smoothed Hz -> steer torque (Nm), power (W), brake (m/s^2)."""
        F = self.F
        z = (f - self.center if self.center is not None else f) / self.rate_scale
        arg = (z * theta[:, :F]).sum(1) + theta[:, F]
        if self.lane_filter is not None:
            arg = arg + theta[:, -1] * self.lane_signal(f)
        steer = self.steer_max * torch.tanh(arg)
        fwd = f[:, self.r.forward].mean(1) / 20.0 if len(self.r.forward) else torch.zeros(f.shape[0], device=f.device)
        power = self.power_max * torch.sigmoid(theta[:, F + 1] * fwd + theta[:, F + 2])
        back = f[:, self.r.mdn].mean(1) / 10.0 if len(self.r.mdn) else torch.zeros(f.shape[0], device=f.device)
        brake = self.brake_max * torch.sigmoid(theta[:, F + 3] * back + theta[:, F + 4])
        return steer, power, brake
