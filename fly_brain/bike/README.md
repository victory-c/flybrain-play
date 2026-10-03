# The fly rides a Tarmac SL9

A fruit fly's whole central nervous system (male CNS v1.0, 166,700 neurons, every synapse) is put in
charge of a Specialized S-Works Tarmac SL9. The bike's state is turned into firing of the fly's real
balance, wind, optic-flow and leg sensory neurons; the brain is simulated; the firing of identified
descending neurons is decoded into handlebar torque, pedalling and braking; the bike moves; repeat
every 10 ms of bike time.

```
        bike state ──► bike/senses.py ──► 789 sensory neurons
                                                │
                                        brain/loop.py  (whole CNS, LIF, 0.1 ms steps)
                                                │
   bike/dynamics.py ◄── bike/readout.py ◄── 92 read-out neurons (33 DN + wing MN types, L/R)
   (Whipple model of the Tarmac, bike/tarmac.py)
```

## What is real and what is modeled

| Piece | Status |
|---|---|
| Neurons, synapses, signs, L/R identity | connectome (Janelia / Google male CNS v1.0) |
| Neuron model and constants | Shiu et al. 2024, synapse scale 0.45 as in the Fly Bar |
| Which sensory neurons carry roll, yaw, lean, wind, handlebar, goal | real cell types (VS/HS cells, halteres, Johnston's organ, front-leg proprioceptors, ORN_DM1); the mapping from bike state to their rates is **modeled** |
| Which descending neurons mean steer / forward / back / escape | literature (DNa01/02/03, DNb01, DNg100, DNp09, DNg97, MDN, DNp01) |
| Decoder: DN rates → torque, power, brake | **learned** (71 parameters: 66 L/R steering weights, a bias, 4 pedal/brake terms; cross-entropy method); the connectome is never changed |
| The bicycle | Whipple–Carvallo linearised model, verified against Meijaard et al. 2007; Tarmac SL9 56 cm geometry, 6.9 kg bike, 70 kg rider (assumed mass distribution) |

"Teaching" here means learning the interface, the way a brain–machine interface decoder is fitted:
the wiring is fixed, and the question is whether the descending neurons carry enough information
about the bike's lean to steer it. If they do not, no decoder will ride, which is a result too.

The Tarmac with a 70 kg rider is self-stable between 4.7 and 7.7 m/s (17 to 28 km/h): above 17 km/h
it rides itself, hands off. Riding starts at 4 m/s, below that range, with 15 Nm side gusts, so the
brain has to do something.

## What the screen found

`runs/screen.py` lets a conventional PD rider steer while the brain only watches, and correlates every
neuron's rate with the bike's lean. Two things came out:

* The walking command neurons of the literature (DNa01, DNa02, DNg100, DNp09...) stay silent, and MDN
  fires (5-19 Hz) without following the lean (|r| < 0.02): balance information does not reach them
  from these senses. What *does* track the roll rate, with opposite sign left and right (|r| up to 0.78), is a set of flight-steering descending neurons
  (DNp20, DNp22, DNg46, DNge043, DNb06, DNp17...) and the wing steering-muscle motor neurons
  (b1, b2, b3, hg1, hi2). Optic flow into VS/HS cells and haltere input reach the central complex
  (ExR, PEN, FB neurons) and those DNs, which is the fly's gaze/flight stabilisation pathway. So the
  fly steers the bike with the reflex it uses to stabilise flight. The decoder reads those neurons.
* A food odour marking the goal (ORN_DM1 at 20 Hz) drives the antennal lobe and mushroom body into
  saturation (APL at 436 Hz, half a million spikes per second) at the Fly Bar's synapse scale, so
  `--goal odor` is off by default. Without it the brain sits at ~35,000 spikes/s and is stable.

The learning starts from a "teacher" fit: ridge regression of the PD rider's torque on the decoder's
features from the screen (`--init results/screen_samples.npz`), then the cross-entropy method refines
the weights in closed loop, where the fly steers on its own.

## Run

```bash
python -m tests.test_bike                        # Whipple matrices vs the paper, SL9 stability range
python -m tests.test_ride                        # bike falls / PD rider holds it; loop on a toy brain
python -m runs.ride --oracle                     # PD rider, no brain
python -m runs.ride --open-loop --riders 8       # brain in the loop, prior decoder (no steering)
python -m runs.screen --riders 16 --seconds 8    # passenger screen -> results/screen*.{json,parquet,npz}
python -m runs.ride --init results/screen_samples.npz --riders 48 --generations 15  # learn
python -m runs.ride --replay results/ride.json   # best decoder, logs state + DN rates every 10 ms
python -m export.export_ride3d results/ride_trace.json results/ride_3d.html   # 3D road replay (chase / side / close-up / fly's eyes)
python -m export.export_ride3d results/ride_trace.json out.html --bike none    # same, procedural Tarmac SL9 instead of the model
python -m export.export_ride results/ride_trace.json results/ride_view.html   # 2D charts replay
```

On the cluster: `sbatch ride.sbatch --init results/screen_samples.npz --riders 48 --generations 15` (repo
root one level up; see JOBS.md there for every run's exact command).

## Lane keeping (staying on the 7 m road)

The lane cue is optic flow on the horizontal-system (HS) cells: drifting or heading toward one road edge
drives that eye's HS cells harder (`--lane hs`, gains `hs_lane` per m, `hs_heading` per rad). Four findings:

* **The balance neurons carry no heading.** In the passenger screen, every roll-rate neuron looked like a
  heading neuron (r_psi about -0.85 r_phi_dot), but that is the bike's weave: heading swings in antiphase with
  roll rate. Flipping which eye's VS/HS cells get the rotation (`--polarity physio`) flipped both together.
* **The fly has a separate yaw/lane channel.** `runs/probe.py` holds the bike upright and drives only the left or
  only the right HS cells: DNa16, DNb03, DNa06, DNp15 (the HS-to-neck DN), DNge107/086/031/033, DNg41, DNp18 and
  the neck motor neuron GNG283 respond, a different set from the VS-driven roll channel (DNp20, DNg46, DNp22...).
  These are `LANE_DNS` (`--readout lane`), read as deviations from each rider's warm-up rate.
* **The lane channel is slow and noisy**, so it is smoothed over 300 ms (`--tau-lane-ms 300`) while roll stays at
  40 ms, and it enters as one matched-filter signal (the probe's response pattern, `--lane-filter results/probe.json`)
  with one learned gain.
* **Test at road speed.** At 5.5 m/s (20 km/h) the Tarmac/V4Rs is self-stable, so a 5 Nm breeze does not knock it
  over but walks it off the road in about 5 s hands-off: staying on the road is the fly's steering.

Result (`ride-slow.sbatch`, then `sweep-gain.sbatch`): the same decoder (the CEM population mean at generation 12,
`results/sweep/`) with only the lane gain changed, 48 riders x 15 s, identical gusts, leaving the road ends a run.

| lane gain | mean time on road | riders on for all 15 s |
|---|---|---|
| hands-off (no steering) | 5.0 s | 0% |
| 0 (lane channel off) | 4.7 s | 2% |
| **-0.29 (learned)** | **8.4 s** | **17%** |
| -0.5 | 6.8 s | 6% |
| -1 / -2 / -4 | 3.7 / 1.7 / 0.8 s | 0% |
| +1 (wrong sign) | 1.2 s | 0% |

Replay of the final decoder (population mean, 16 riders x 20 s, no cut-off): 9.1 s on the road on average
(hands-off: 5.4 s), 2 riders on it the whole 20 s. Too much gain over-corrects through the brain's lag, which is
why the optimum is small. Things that did not work: CEM with a lane penalty only (balanced, but headings
wandered 60-280 deg), DAgger imitation of the PD rider (lag makes the copied steering unstable, ~4.5 s upright).

```bash
python -m runs.probe --pop HS VS HALT --out results/probe.json
python -m runs.ride --replay results/ride_slow_mu.json --tau-lane-ms 300 --lane-filter results/probe.json \
    --readout lane --lane hs --gains hs_heading=150,hs_lane=50 --v0 5.5 --gust 5 --riders 16 --seconds 20 \
    --trace results/ride_slow_mu_trace.json
python -m export.export_ride3d results/ride_slow_mu_trace.json results/ride_3d.html
```

## The 3D bike in the replay

The followed rider rides `assets/colnago_v4rs.glb`: the Colnago V4Rs (size 510, Campagnolo Super
Record, Bora Ultra WTO) that Colnago's own web configurator loads from its CDN
(assets.v2.londondynamics.com, pre-assembled "puzzledefault" GLB, 26 MB), Draco-compressed to 5.7 MB
and embedded in the HTML. No free, downloadable model of a Tarmac SL9, Cervelo S5 or Giant TCR
exists (a sweep of Sketchfab, Objaverse, GitHub, print sites, marketplaces, game-asset libraries and
brand configurators found only paid ones, $20-95 on CGTrader, TurboSquid and 3DModels.org).

It is Colnago's copyrighted marketing asset. The GLB itself is kept out of git (`assets/*.glb` is
ignored), but the 3D replay pages embed it, and those pages are public on flybrain-play.vercel.app; if
that has to stop, rebuild them with `--bike none` (procedural bike). The page re-parents the model's parts onto
pivots: fork, integrated bar/stem and front wheel steer about the head-tube axis (72 deg, which puts
the front hub 48.6 mm, the fork rake, ahead of the axis through the headset cap); both wheels with
their rotors and the crank arms with chainrings spin. Saddle, hoods, bottom bracket and crank angle
are read off the model, and the rider is sized to them.

## The brain map in the replay

The left panel of the 3D page shows the followed rider's whole brain while it rides: the same point cloud
(140,638 neurons with a 3D position) and the same regions as the drinks dashboard (Fly Brain Live), every
100 ms. `--brain-out` makes `runs.ride --replay` record every rider's spikes per bin
(`results/<name>_brain.npz`, git-ignored, ~0.5 GB uncompressed for 16 riders x 20 s); the exporter picks it
up next to the trace and embeds the default rider neuron by neuron and every rider's region means (other
riders are coloured by region). Region means leave out the sensory neurons the bike drives.

The page's run (recorded on CPU while the GPUs were busy, so its noise differs from the GPU replay above) gives
8.1 s on the road on average and 1 of 16 riders on it for all 20 s; hands-off under the same conditions and seed,
5.2 s and none (`results/ride_pilot_road_trace.json`, the open-loop page).

```bash
python -m runs.ride --replay results/ride_slow_mu.json --tau-lane-ms 300 --lane-filter results/probe.json \
    --readout lane --lane hs --gains hs_heading=150,hs_lane=50 --v0 5.5 --gust 5 --riders 16 --seconds 20 --seed 7 \
    --trace results/ride_road_trace.json --brain-out results/ride_road_brain.npz
python -m export.export_ride3d results/ride_road_trace.json results/ride_3d.html
```

Speed on one RTX A6000: 48 riders run about 20 s of wall time per second of bike time, so a 10 s
generation takes ~3.4 min and a 15-generation search ~51 min (logs/flyride-36091, -36092); a 16-rider
replay runs at ~13.4 s per second of bike time.
