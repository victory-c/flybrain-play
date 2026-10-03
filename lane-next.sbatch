#!/bin/bash
#SBATCH --job-name=flylane
#SBATCH --partition=ocf-hpc
#SBATCH --nodelist=corruption
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:20:00
#SBATCH --output=/home/s/st/stevejobs/flybrain/logs/flylane-%j.out
#SBATCH --error=/home/s/st/stevejobs/flybrain/logs/flylane-%j.err
# Lane keeping, next round (~3 GPU h, one job, sequential):
#  1) four 48 x 15 s replays on seed 31 of the sweep decoder (gen-12 mean, results/sweep/gain_-0.29.json: 8.40 s on the
#     road), paired rider by rider with results/sweep/*.trace.json: batch-mean centring, then three cue/gain variants;
#  2) CEM from ride_slow_mu under the winner (96 riders x 12 s, bias searched at the riding operating point, pedal and
#     brake frozen) until the budget is used up;
#  3) held-out check on seed 59: the CEM mean vs ride_slow_mu under the old protocol (and ride_slow_mu untrained under
#     the new one). Adopt the new decoder only if it beats the old by >= 1.5 s on the road (about 2 paired SE).
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /home/s/st/stevejobs/flybrain/fly_brain
PY=${PY:-/home/s/st/stevejobs/flybrain/venv-cuda/bin/python}
D=${OUTDIR:-results/lane}; BUDGET_MIN=${BUDGET_MIN:-178}; T0=$(date +%s)
mkdir -p $D
BASE="--tau-lane-ms 300 --lane-filter results/probe.json --readout lane --lane hs --v0 5.5 --gust 5 --offroad 3.5"
ontime() { grep -oP '^replay of .*: upright +\K[0-9.]+' $D/$1.log || echo 0; }   # with --offroad: s upright AND on the road
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a > b)}'; }
replay() {  # name decoder seed [flags]
  local name=$1 dec=$2 seed=$3; shift 3
  echo "== $(date +%T) $name: $dec seed $seed $*"
  $PY -m runs.ride --replay $dec $BASE --riders 48 --seconds 15 --seed $seed "$@" --trace $D/$name.trace.json > $D/$name.log 2>&1 || true
  grep -E "^replay|Error" $D/$name.log || true
}

# 0) CPU: zbar_mu.npy (mean input of each decoder weight while riding, from ride_slow_mu's seed-7 trace) and S12 at gain -0.15
$PY - $D <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
from bike.readout import LANE_DNS
out = Path(sys.argv[1])
d = json.load(open("results/ride_slow_mu_trace.json"))          # ride_slow_mu, 16 riders x 20 s, seed 7
names, tn = d["dn_names"], d["theta_names"]
assert tn == json.load(open("results/ride_slow_mu.json"))["decoder_names"]
X = np.array([r["dn_hz"] for r in d["trace"]], np.float32)      # (T, B, F) 40 ms-smoothed Hz
S = np.array([r["state"] for r in d["trace"]], np.float32)
A = ~np.array([r["done"] for r in d["trace"]]) & (np.abs(S[:, :, 1]) < 3.5)   # alive and on the road
f0 = X[0].mean(0)                                               # ~ batch-mean warm-up rate (= --center-batch)
cen = np.array([n.rsplit("_", 1)[0] in LANE_DNS for n in names])
by = {}
for r in json.load(open("results/probe.json"))["HS"]:           # = runs.ride.lane_filter_from_probe
    if abs(r["z"]) >= 3.0: by.setdefault((r["type"], r["somaSide"]), []).append(r["diff"])
v = np.array([np.mean(by[tuple(n.rsplit("_", 1))]) if tuple(n.rsplit("_", 1)) in by else 0.0 for n in names], np.float32)
w = v / max(float((v ** 2).sum()), 1e-6)
fbar, F = X[A].mean(0), len(names)
zb = np.zeros(len(tn), np.float32)
zb[:F] = np.where(cen, fbar - f0, fbar) / 50.0                  # input of each steering weight (rate_scale 50)
zb[tn.index("steer:lane_filter")] = (fbar - f0) @ w             # mean lane signal while riding
np.save(out / "zbar_mu.npy", zb)
print(f"zbar: {A.sum()} on-road samples, |z| uncentred {np.linalg.norm(zb[:F][~cen]):.2f}, lane signal {zb[-1]:+.3f}")
# S12 at lane gain -0.15; the lane signal's closed-loop offset (+0.73 in results/sweep/gain_-0.29.trace.json) moves into steer:bias
src = json.load(open("results/sweep/gain_-0.29.json")); t = list(src["best_theta"]); n = src["decoder_names"]
ib, ig = n.index("steer:bias"), n.index("steer:lane_filter")
t[ib] -= (-0.15 - t[ig]) * 0.73; t[ig] = -0.15
json.dump({"best_theta": t, "decoder_names": n, "note": "S12, lane gain -0.15, offset moved into steer:bias"}, open(out / "s12_g-0.15.json", "w"))
EOF

# 1) seed-31 replays of S12, paired with the sweep (gain -0.29, per-rider centring: 8.40 s on the road)
S12=results/sweep/gain_-0.29.json
replay c31_cb $S12 31 --center-batch --gains hs_heading=150,hs_lane=50
CB="--center-batch"; T_cur=$(ontime c31_cb)
if gt 7.70 $T_cur; then echo "-> batch centring rejected ($T_cur s)"; CB=""; T_cur=8.40; fi
replay c31_L10 $S12 31 $CB --gains hs_heading=150,hs_lane=15                    # 10 m look-ahead, same heading gain
replay c31_h450 $D/s12_g-0.15.json 31 $CB --gains hs_heading=450,hs_lane=50     # 9 m look-ahead, 3x HS heading drive
replay c31_g15 $D/s12_g-0.15.json 31 $CB --gains hs_heading=150,hs_lane=50      # control for h450: same gain, old cue
pick=cur; T_pick=$T_cur
for c in L10 h450 g15; do   # a variant must beat the current cue by 0.7 s (~1 paired SE); then the best one wins
  t=$(ontime c31_$c); thr=$(awk -v a=$T_pick -v p=$pick 'BEGIN{print (p == "cur") ? a + 0.7 : a}')
  if gt $t $thr; then pick=$c; T_pick=$t; fi
done
case $pick in
  cur)  GAINS=hs_heading=150,hs_lane=50; G=keep ;;
  L10)  GAINS=hs_heading=150,hs_lane=15; G=keep ;;
  h450) GAINS=hs_heading=450,hs_lane=50; G=-0.15 ;;
  g15)  GAINS=hs_heading=150,hs_lane=50; G=-0.15 ;;
esac
echo "-> seed 31: cur $T_cur, L10 $(ontime c31_L10), h450 $(ontime c31_h450), g15 $(ontime c31_g15) s; pick $pick: --gains $GAINS, lane gain $G, '$CB'"
$PY - $G $D <<'EOF'
import json, sys
import numpy as np
g, out = sys.argv[1], sys.argv[2]
src = json.load(open("results/ride_slow_mu.json")); t = list(src["best_theta"]); n = src["decoder_names"]
if g != "keep":   # new lane gain; its riding offset (zbar) moves into steer:bias so the standing torque is unchanged
    zb = np.load(f"{out}/zbar_mu.npy"); ib, ig = n.index("steer:bias"), n.index("steer:lane_filter")
    t[ib] -= (float(g) - t[ig]) * float(zb[ig]); t[ig] = float(g)
json.dump({"best_theta": t, "decoder_names": n, "note": f"ride_slow_mu, lane gain {g}"}, open(f"{out}/init.json", "w"))
EOF

# 2) CEM until the budget leaves 17 min for the three seed-59 replays (48 riders if 96 fails to start)
cem() {  # riders
  local left=$(( BUDGET_MIN - ($(date +%s) - T0) / 60 - 17 ))
  echo "== $(date +%T) CEM, $1 riders, up to $left min"
  $PY -m runs.ride --init-theta $D/init.json --zbar $D/zbar_mu.npy --sigma-pedal 0 $CB $BASE --gains $GAINS \
    --sigma0 0.05 --sigma-lane-filter 0.08 --sigma-min 0.03 --lane-penalty 0.02 --riders $1 --seconds 12 \
    --generations 40 --max-minutes $left --seed 41 --out $D/cem.json 2>&1 | tee $D/cem.log | grep -E "^gen|^done|^stopping|starts|Error" || true
}
rm -f $D/cem.json
cem 96
[ -s $D/cem.json ] || cem 48
$PY -c "import json; d = json.load(open('$D/cem.json')); json.dump({'best_theta': d['mu'], 'decoder_names': d['decoder_names'], 'note': 'CEM mean after %d generations' % len(d['history'])}, open('$D/cem_mu.json', 'w'))" || cp $D/init.json $D/cem_mu.json

# 3) held-out seed 59, 48 x 15 s, paired by rider
replay v59_cem $D/cem_mu.json 59 $CB --gains $GAINS
replay v59_mu_old results/ride_slow_mu.json 59 --gains hs_heading=150,hs_lane=50
replay v59_init $D/init.json 59 $CB --gains $GAINS
echo "== seed 59 on the road: CEM $(ontime v59_cem) s, ride_slow_mu old protocol $(ontime v59_mu_old) s, ride_slow_mu new protocol $(ontime v59_init) s"
echo "== done $(date)"
