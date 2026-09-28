"""What the fly senses on the bike, and which real neurons of the male CNS carry it.

Every population below is a set of identified sensory neurons in the connectome; the *encoding*
(bike state -> firing rate) is a MODELED assumption, kept simple, monotone and side-specific:

  population        neurons (male CNS)                      carries                      side rule
  VS_L / VS_R       VS, VSm, VST1, VST2 lobula-plate cells   roll rate (roll optic flow)  side the bike rolls toward
  HS_L / HS_R       HSN, HSE, HSS, HST, H2                   forward flow + yaw rate      side the bike yaws toward
  HALT_L / HALT_R   haltere afferents (DMetaN)               |roll rate|, |yaw rate|      roll toward that side
  JO_L / JO_R       Johnston's organ wind/gravity (JO-C/E)   airspeed, lean angle          lean toward that side
  LEG1_L / LEG1_R   front-leg proprioceptors (ProLN)         handlebar angle and rate     bar turned away from that side
  ORN_L / ORN_R     ORN_DM1 (a food odour)                   bearing of a goal 50 m ahead the side the goal is on
  (lane="hs")       HS cells again                           lateral offset + heading   side of the nearer road edge

Sides are the connectome's rootSide for neurons that enter through a nerve (halteres, JO, legs,
antennae) and somaSide for the optic-lobe cells. Rates are Hz, clipped to [0, cap].
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as ft
import torch

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

GAINS = {  # Hz per unit
    "base": 2.0, "cap": 200.0,
    "vs_roll": 100.0,       # per rad/s
    "hs_speed": 4.0,        # per m/s (progressive flow, both eyes)
    "hs_yaw": 60.0,         # per rad/s
    "halt_roll": 120.0,     # per rad/s
    "halt_yaw": 60.0,       # per rad/s
    "jo_wind": 6.0,         # per m/s airspeed
    "jo_lean": 150.0,       # per rad of lean
    "leg_steer": 200.0,     # per rad of handlebar
    "leg_steer_rate": 50.0, # per rad/s
    "orn_goal": 40.0,       # at full bearing (>= 0.4 rad to one side)
    "goal_ahead": 50.0,     # m
    "hs_lane": 25.0,        # Hz per m of lateral offset (translational flow from the nearer road edge)
    "hs_heading": 60.0,     # Hz per rad of heading away from the road direction
}


def with_sides(meta):
    """brain_meta plus rootSide / entryNerve from the raw annotation table."""
    raw = ft.read_table(ANNOTATIONS, columns=["bodyId", "rootSide", "entryNerve"]).to_pandas()
    return meta.merge(raw, on="bodyId", how="left")


def sensory_populations(meta, goal="none", lane="none"):
    """OrderedDict name -> neuron indices (into brain.npz order)."""
    m = with_sides(meta) if "rootSide" not in meta.columns else meta
    ty, sub, cl = m["type"].astype(str), m["subclass"].astype(str), m["class"].astype(str)
    soma, root = m["somaSide"].astype(str), m["rootSide"].astype(str)
    pops = OrderedDict()

    def put(name, mask):
        pops[name] = m.loc[mask, "idx"].to_numpy(dtype=np.int64)

    vs = ty.isin(["VS", "VSm", "VST1", "VST2"])
    hs = ty.isin(["HSN", "HSE", "HSS", "HST", "H2"])
    halt = sub == "haltere"
    jo = sub == "wind_gravity"
    leg1 = (cl == "mechanosensory_proprioceptive") & (m["entryNerve"] == "ProLN")
    orn = ty == "ORN_DM1"
    for side in ("L", "R"):
        put(f"VS_{side}", vs & (soma == side))
        put(f"HS_{side}", hs & (soma == side))
        put(f"HALT_{side}", halt & (root == side))
        put(f"JO_{side}", jo & (root == side))
        put(f"LEG1_{side}", leg1 & (root == side))
        if goal == "odor":
            put(f"ORN_{side}", orn & (root == side))
    return pops


class Senses:
    def __init__(self, meta, goal="none", gains=None, device="cpu", lane="none", polarity="legacy"):
        """polarity: which eye's VS/HS cells a self-rotation drives.
        'physio' follows fly lobula-plate physiology (Hausen 1982; Krapp & Hengstenberg 1996; Joesch et al. 2008):
          HS cells are depolarised by front-to-back motion in their own eye, so a yaw to the RIGHT excites HS_L;
          VS cells are depolarised by downward motion, so a roll to the RIGHT (right side down; the right eye sees the
          world move up, the left eye sees it move down) excites VS_L.
        'legacy' is the first, arbitrary assignment (the rotation's own side), kept so older decoders replay."""
        self.g = dict(GAINS, **(gains or {}))
        self.goal, self.lane, self.polarity = goal, lane, polarity
        self.pops = sensory_populations(meta, goal)
        self.names = list(self.pops)
        self.idx = np.concatenate([self.pops[n] for n in self.names])
        sizes = [len(self.pops[n]) for n in self.names]
        self.sizes = dict(zip(self.names, sizes))
        self.dev = torch.device(device)
        # expansion matrix: population rate (B, P) -> neuron rate (B, n_stim)
        E = torch.zeros(len(self.names), len(self.idx), device=self.dev)
        o = 0
        for i, n in enumerate(sizes):
            E[i, o:o + n] = 1.0
            o += n
        self.E = E

    def describe(self):
        return ", ".join(f"{n} {s}" for n, s in self.sizes.items()) + f"  ({len(self.idx)} neurons)"

    @torch.no_grad()
    def population_rates(self, state, psi_dot):
        """state (B, 8) from bike.dynamics, psi_dot (B,) -> (B, P) Hz per population."""
        g = self.g
        relu = torch.relu
        y, psi, v, phi, delta, phi_dot, delta_dot = (state[:, i] for i in (1, 2, 3, 4, 5, 6, 7))
        base = g["base"]
        out = {}
        s = 1.0 if self.polarity == "physio" else -1.0   # physio: right rotations drive the LEFT cells
        out["VS_L"] = base + g["vs_roll"] * relu(s * phi_dot)
        out["VS_R"] = base + g["vs_roll"] * relu(-s * phi_dot)
        out["HS_L"] = base + g["hs_speed"] * v + g["hs_yaw"] * relu(s * psi_dot)
        out["HS_R"] = base + g["hs_speed"] * v + g["hs_yaw"] * relu(-s * psi_dot)
        if self.lane == "hs":
            # MODELED lane cue on the horizontal-system cells: drifting right (y > 0) brings the right road edge
            # closer, so the right eye sees stronger flow; heading right of the road (psi > 0) likewise.
            out["HS_R"] = out["HS_R"] + g["hs_lane"] * relu(y) + g["hs_heading"] * relu(psi)
            out["HS_L"] = out["HS_L"] + g["hs_lane"] * relu(-y) + g["hs_heading"] * relu(-psi)
        out["HALT_L"] = base + g["halt_roll"] * relu(-phi_dot) + g["halt_yaw"] * psi_dot.abs()
        out["HALT_R"] = base + g["halt_roll"] * relu(phi_dot) + g["halt_yaw"] * psi_dot.abs()
        out["JO_L"] = base + g["jo_wind"] * v + g["jo_lean"] * relu(-phi)
        out["JO_R"] = base + g["jo_wind"] * v + g["jo_lean"] * relu(phi)
        out["LEG1_L"] = base + g["leg_steer"] * relu(delta) + g["leg_steer_rate"] * relu(delta_dot)
        out["LEG1_R"] = base + g["leg_steer"] * relu(-delta) + g["leg_steer_rate"] * relu(-delta_dot)
        if self.goal == "odor":
            bearing = torch.atan2(-y, torch.full_like(y, g["goal_ahead"])) - psi  # < 0: goal to the left
            side = (bearing / 0.4).clamp(-1.0, 1.0)
            out["ORN_L"] = g["orn_goal"] * (0.5 - 0.5 * side)
            out["ORN_R"] = g["orn_goal"] * (0.5 + 0.5 * side)
        P = torch.stack([out[n] for n in self.names], 1)
        return P.clamp(0.0, g["cap"])

    def rates(self, state, psi_dot):
        """(B, n_stim) Hz for BrainLoop.set_rates."""
        return self.population_rates(state, psi_dot) @ self.E
