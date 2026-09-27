"""Time-resolved whole-brain replays for the activity dashboard -> dashboard/data/.

For each drink: simulate all trials, keep every neuron's spikes in 50 ms bins, and write
  <id>.bin    uint8 (n_bins, n_points) firing rate of each 3D cloud point, 255 = CLOUD_HZ
  <id>.json   traces: MN9 L/R, taste channels, brain regions, population; taste + recipe
plus index.json listing drinks, regions and bins. brain_xyz.bin / brain_group.bin are copied.

usage: python -m export.export_dashboard [--trials 8] [--only 橙汁,奶茶]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from brain.sim import W_SYN_MALE_CNS, Brain, simulate
from brain.taste import CHANNELS, DRINKS, COCKTAILS, INGREDIENTS, neuron_input, taste_matrix, taste_vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "data"
BIN_MS = 50.0
CLOUD_HZ = 40.0

# (id, display name, drink vector, hunger)
MENU = [
    ("water",      "💧 清水",          DRINKS["💧 Νερό"], 0.0),
    ("orange",     "🍊 橙汁",          DRINKS["🍊 Χυμός πορτοκάλι"], 0.0),
    ("watermelon", "🍉 西瓜汁",        np.array([60, 0, 0, 0, 0, 0, 5.5], float), 0.0),
    ("milktea",    "🧋 珍珠奶茶",      np.array([80, 0, 150, 0, 0, 0, 6.5], float), 0.0),
    ("lemonade",   "🍋 柠檬水",        DRINKS["🍋 Λεμονάδα"], 0.0),
    ("cola",       "🥤 可乐",          DRINKS["🥤 Κόλα"], 0.0),
    ("espresso",   "☕ 浓缩咖啡",      DRINKS["☕ Espresso"], 0.0),
    ("lager",      "🍺 拉格啤酒",      DRINKS["🍺 Μπίρα λάγκερ"], 0.0),
    ("tsipouro",   "🥃 茨普罗烈酒",    DRINKS["🥃 Τσίπουρο"], 0.0),
    ("pinacolada", "🍍 Piña Colada",   COCKTAILS["🍍 Piña Colada"], 0.0),
    ("pinacolada_hungry", "🍍 Piña Colada（饿）", COCKTAILS["🍍 Piña Colada"], 0.5),
    ("negroni_hungry",    "🥃 Negroni（很饿）",   COCKTAILS["🥃 Negroni"], 0.9),
]

REGIONS = [  # (id, label, colour hint) in display order
    ("taste", "味觉神经元"), ("feeding", "摄食中枢 SEZ"), ("smell", "嗅觉"), ("memory", "记忆与奖赏 (蘑菇体)"),
    ("nav", "导航 (中央复合体)"), ("vision", "视觉"), ("touch", "触觉与本体感觉"), ("descending", "下行指令"),
    ("cord", "腹神经索 (腿/翅)"), ("motor", "运动神经元"), ("other", "中央脑其他"),
]


def region_of(t, c, s):
    t, c, s = (x if isinstance(x, str) else "" for x in (t, c, s))
    if c == "gustatory":
        return "taste"
    if c in ("olfactory", "ALPN", "ALLN", "ALIN", "ALON"):
        return "smell"
    if c in ("Kenyon_Cell", "MBON", "DAN"):
        return "memory"
    if c == "CX":
        return "nav"
    if s.startswith(("ol_", "visual")) or c == "visual":
        return "vision"
    if "motor" in s or "efferent" in s:
        return "motor"
    if s == "descending_neuron":
        return "descending"
    if "mechanosensory" in c or "proprio" in c:
        return "touch"
    if s.startswith("vnc") or s in ("ascending_neuron", "sensory_ascending"):
        return "cord"
    if t.startswith(("GNG", "PRW", "SEZ")) or c == "SEZPN":
        return "feeding"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--ms", type=float, default=1000.0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("brain_xyz.bin", "brain_group.bin"):
        shutil.copy(ROOT / "web_data" / f, OUT / f)
    point_idx = np.fromfile(ROOT / "web_data" / "brain_idx.bin", dtype=np.int32)

    brain = Brain(ROOT / "brain.npz")
    meta = pd.read_parquet(ROOT / "brain_meta.parquet").sort_values("idx")
    by_instance = meta.set_index("instance")["idx"]
    mn9 = [int(by_instance["MN9_L"]), int(by_instance["MN9_R"])]
    reg = np.array([region_of(t, c, s) for t, c, s in zip(meta["type"], meta["class"], meta["superclass"])])
    reg_ids = [r for r, _ in REGIONS]
    reg_masks = {r: reg == r for r in reg_ids}
    grn_idx, M, organ = taste_matrix(meta)
    n_bins = int(round(args.ms / BIN_MS))
    all_idx = np.arange(brain.n)
    index = {"binMs": BIN_MS, "bins": n_bins, "trials": args.trials, "cloudHz": CLOUD_HZ, "points": int(len(point_idx)),
             "neurons": int(brain.n), "regions": [{"id": r, "label": l, "n": int(reg_masks[r].sum())} for r, l in REGIONS],
             "pointRegion": None, "drinks": []}
    reg_code = np.array([reg_ids.index(r) for r in reg], dtype=np.uint8)
    reg_code[point_idx].tofile(OUT / "point_region.bin")
    index["pointRegion"] = "point_region.bin"

    for did, name, x, hunger in MENU:
        if only and not any(s in name or s in did for s in only):
            continue
        t = taste_vector(x)
        t[CHANNELS.index("bitter")] *= 1.0 - hunger
        u = neuron_input(t, M)
        on = u >= 1.0
        print(f"== {name}  hunger {hunger}  taste " + " ".join(f"{c} {v:.2f}" for c, v in zip(CHANNELS, t)), flush=True)
        r = simulate(brain, grn_idx[on], u[on], readout_idx=all_idx, n_run=args.trials, t_run=args.ms,
                     bin_ms=BIN_MS, params={"w_syn": W_SYN_MALE_CNS}, progress=False)
        hz = r["readout"].mean(0) / (BIN_MS / 1000.0)  # (N, bins) Hz
        cloud = np.clip(hz[point_idx].T / CLOUD_HZ * 255, 0, 255).astype(np.uint8)  # (bins, points)
        cloud.tofile(OUT / f"{did}.bin")
        stim = np.zeros(brain.n, bool)
        stim[grn_idx[on]] = True
        chan = {}
        for j, c in enumerate(CHANNELS):
            members = grn_idx[(M[:, j] > 0) & on]
            chan[c] = hz[members].mean(0).round(1).tolist() if len(members) else [0.0] * n_bins
        regions = {rid: hz[reg_masks[rid] & ~stim].mean(0).round(2).tolist() for rid in reg_ids}
        active = {rid: (hz[reg_masks[rid] & ~stim] > 10).sum(0).astype(int).tolist() for rid in reg_ids}
        per_trial = r["readout"][:, mn9[0]].sum(1) / (args.ms / 1000.0)
        lit_types = meta.loc[(hz.mean(1) > 10) & ~stim & (reg != "vision")].groupby("type").size().sort_values(ascending=False)
        rec = {
            "id": did, "name": name, "hunger": hunger,
            "recipe": dict(zip(INGREDIENTS, x.round(2).tolist())),
            "taste": dict(zip(CHANNELS, t.round(3).tolist())),
            "tasteNeurons": int(on.sum()),
            "mn9Hz": float(per_trial.mean()), "mn9Trials": per_trial.round(0).tolist(),
            "mn9": {"L": hz[mn9[0]].round(1).tolist(), "R": hz[mn9[1]].round(1).tolist()},
            "channels": chan, "regions": regions, "activeCount": active,
            "pop": (r["pop_hz"] / 1000.0).round(1).tolist(),  # thousand spikes per second
            "lit": int(((hz.mean(1) > 2) & ~stim).sum()),
            "topTypes": [{"type": k, "n": int(v)} for k, v in lit_types.head(10).items()],
            "seconds": round(r["seconds"], 1), "file": f"{did}.bin",
        }
        (OUT / f"{did}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        index["drinks"].append({k: rec[k] for k in ("id", "name", "hunger", "mn9Hz", "lit", "file")})
        print(f"   MN9_L {rec['mn9Hz']:.1f} Hz   lit {rec['lit']:,}   {rec['seconds']}s", flush=True)
        (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
