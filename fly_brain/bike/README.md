# The fly rides a Tarmac SL9

A fruit fly's whole central nervous system (male CNS v1.0, 166,700 neurons, every synapse) is put in
charge of a Specialized S-Works Tarmac SL9. The bike's state is turned into firing of the fly's real
balance, wind, optic-flow and leg sensory neurons; the brain is simulated; the firing of identified
descending neurons is decoded into handlebar torque, pedalling and braking; the bike moves; repeat
every 10 ms of bike time.

```
        bike state ──► bike/senses.py ──► 863 sensory neurons
                                                │
                                        brain/loop.py  (whole CNS, LIF, 0.1 ms steps)
                                                │
   bike/dynamics.py ◄── bike/readout.py ◄── 26 descending neurons (12 types, L/R)
   (Whipple model of the Tarmac, bike/tarmac.py)
```

## What is real and what is modeled

| Piece | Status |
|---|---|
| Neurons, synapses, signs, L/R identity | connectome (Janelia / Google male CNS v1.0) |
| Neuron model and constants | Shiu et al. 2024, synapse scale 0.45 as in the Fly Bar |
| Which sensory neurons carry roll, yaw, lean, wind, handlebar, goal | real cell types (VS/HS cells, halteres, Johnston's organ, front-leg proprioceptors, ORN_DM1); the mapping from bike state to their rates is **modeled** |
| Which descending neurons mean steer / forward / back / escape | literature (DNa01/02/03, DNb01, DNg100, DNp09, DNg97, MDN, DNp01) |
| Decoder: DN rates → torque, power, brake | **learned** (29 weights, cross-entropy method); the connectome is never changed |
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

* The walking command neurons of the literature (DNa01, DNa02, DNg100, DNp09, MDN...) stay silent:
  balance information does not reach them from these senses. What *does* track the roll rate, with
  opposite sign left and right (|r| up to 0.76), is a set of flight-steering descending neurons
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
python -m export.export_ride3d results/ride_trace.json results/ride_3d.html   # 3D road replay (chase / side / fly's eyes)
python -m export.export_ride results/ride_trace.json results/ride_view.html   # 2D charts replay
```

On the cluster: `sbatch ride.sbatch --riders 32 --generations 12` (repo root one level up).

Speed on one RTX A6000: 32 brains run about 14 s of wall time per second of bike time; a 10 s
episode for 32 riders is ~2.5 min, so a 12-generation search is ~30 min.
