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

## The 3D bike in the replay

The followed rider rides `assets/colnago_v4rs.glb`: the Colnago V4Rs (size 510, Campagnolo Super
Record, Bora Ultra WTO) that Colnago's own web configurator loads from its CDN
(assets.v2.londondynamics.com, pre-assembled "puzzledefault" GLB, 26 MB), Draco-compressed to 5.7 MB
and embedded in the HTML. No free, downloadable model of a Tarmac SL9, Cervelo S5 or Giant TCR
exists (a sweep of Sketchfab, Objaverse, GitHub, print sites, marketplaces, game-asset libraries and
brand configurators found only paid ones, $20-95 on CGTrader, TurboSquid and 3DModels.org).

It is Colnago's copyrighted marketing asset: fine to look at in a private replay, not to publish or
redistribute, which is why `assets/*.glb` is git-ignored. The page re-parents the model's parts onto
pivots: fork, integrated bar/stem and front wheel steer about the head-tube axis (72 deg, which puts
the front hub 48.6 mm, the fork rake, ahead of the axis through the headset cap); both wheels with
their rotors and the crank arms with chainrings spin. Saddle, hoods, bottom bracket and crank angle
are read off the model, and the rider is sized to them.

Speed on one RTX A6000: 48 riders run about 20 s of wall time per second of bike time, so a 10 s
generation takes ~3.4 min and a 15-generation search ~51 min (logs/flyride-36091, -36092); a 16-rider
replay runs at ~13.4 s per second of bike time.
