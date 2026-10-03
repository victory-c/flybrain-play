# flybrain-play

A fruit fly's whole central nervous system, 166,700 neurons and every synapse between them, run as
a spiking network. We give it drinks, record every neuron, and put it in charge of a road bike.

**Live demos: https://flybrain-play.vercel.app**

| Demo | What you see |
|---|---|
| [The fly rides a Colnago V4Rs](https://flybrain-play.vercel.app/ride/) | 3D replay: the brain rides a Colnago V4Rs down a 7 m road at 20 km/h in side gusts, balancing with its flight-stabilising descending neurons and keeping to the road with its HS optic-flow cells. Chase cam, side cam, or the fly's own eyes; the side panel shows his whole brain firing, region by region, as in Fly Brain Live. Also: [charts](https://flybrain-play.vercel.app/ride/charts), [PD rider without a brain](https://flybrain-play.vercel.app/ride/oracle), [brain in the loop but not steering](https://flybrain-play.vercel.app/ride/open-loop) |
| [Fly Brain Live](https://flybrain-play.vercel.app/dashboard/) | Brain activity in 50 ms frames (all 140,638 neurons with a known position) across 12 replays of 11 drinks (orange juice, bubble tea, cola, espresso, lager, tsipouro, ...; Piña Colada fed and hungry, Negroni very hungry), with the feeding motor neuron MN9 |
| [Fly Bar](https://flybrain-play.vercel.app/bar/) | The 3D bar app: nine cocktails ranked by how much the fly wants them, and the brain lighting up |
| [Fly Bar, classic](https://flybrain-play.vercel.app/classic) | Single-page version: from the tongue to the proboscis |

## What is in here

The simulator is [sstamou03/fly_brain](https://github.com/sstamou03/fly_brain) ("Fly Bar"): a leaky
integrate-and-fire network over the male CNS v1.0 connectome (Janelia / Google, 2026), neuron model
and constants after Shiu et al., *Nature* 2024. The first commit of this repo is that project,
unmodified; everything after it is ours:

1. **Ran the Fly Bar on a Slurm cluster** (`flybar.sbatch`, `serve.sh`, `fly_brain/runs/serve.py`).
   `./serve.sh "🧋 奶茶" --sugar 80 --caffeine 150 --ph 6.5` serves any custom drink.
2. **Fly Brain Live dashboard** (`fly_brain/dashboard/`, `fly_brain/export/export_dashboard.py`,
   `pack_dashboard.py`): time-resolved replays of the whole 3D neuron cloud (140,638 neurons with a known position) for 12
   replays of 11 drinks.
3. **GPU port** (`fly_brain/brain/sim.py`, `device="cuda"`; `runs/bench_gpu.py`). A single trial is
   Python-loop bound (~8.5 s per simulated second either way); batching wins: 128 trials in 34.6 s on
   one RTX A6000.
4. **The fly rides a bike** (`fly_brain/bike/`, `brain/loop.py`, `runs/screen.py`, `runs/ride.py`,
   `ride.sbatch`). Details in [fly_brain/bike/README.md](fly_brain/bike/README.md). In short:
   - bike state is turned into firing of real balance, wind, optic-flow and leg sensory neurons;
     descending-neuron firing is decoded into steering torque, pedalling and braking;
   - the walking command neurons of the literature do not track the lean (most stay silent; MDN fires
     at 5-19 Hz regardless of it), but flight-steering descending neurons (DNp20, DNg46, DNp22, ...)
     track roll rate with opposite sign left and right (DNp20 r = +0.78 / -0.74), so the fly steers
     the bike with its flight-stabilisation reflex;
   - a 71-parameter decoder (66 steering weights on 33 descending- and wing-motor-neuron types, left
     and right, plus a bias and 4 pedal/brake terms) fitted to a PD teacher (R² 0.75) and refined
     with the cross-entropy method keeps 16 riders up for a mean of 14.1 s out of 15 s; 9 of 16 never
     fall (a 10th goes down at 14.99 s). The connectome is never changed.
5. **Demo site** (`site/`, `vercel.json`): `site/build.sh` builds the app and collects the pages.

```
flybrain-play/
├── fly_brain/        the simulator + our additions (brain/, runs/, bike/, export/, dashboard/, app/, results/)
├── site/             landing page and build script for the Vercel site
├── *.sbatch          Slurm jobs: flybar (CPU), cuda-venv, bench, dashboard, ride, ride-lane (GPU)
├── serve.sh          serve one drink via srun
├── sync.sh           commit + push everything that changed (code, results, logs, JOBS.md)
├── tools/            jobs_ledger.py writes JOBS.md from sacct
├── JOBS.md           every Slurm job (sbatch and srun) with its exact command line and outputs
└── logs/             Slurm output files of the sbatch jobs; logs/inline/ has scripts fed to srun on stdin
```

## Running it

The web demos need nothing but a browser. To rebuild the site locally (Node 20.19+ or 22.12+, as Vite 7 requires):

```bash
bash site/build.sh && npx serve site/dist
```

To run the simulations you need Python 3.11+ with `numpy pandas pyarrow torch` and the male CNS v1.0
files (~1.1 GB, [male-cns.janelia.org/download](https://male-cns.janelia.org/download)) in
`fly_brain/data/`; `cd fly_brain && python -m brain.build_brain` turns them into `fly_brain/brain.npz`
(run every `python -m ...` step from `fly_brain/`). The sbatch scripts and `serve.sh` are written for the
OCF `corruption` node (partition `ocf-hpc`) with absolute `/home/s/st/stevejobs/flybrain` paths; for
another cluster, edit the `#SBATCH` lines, `ROOT` in flybar/cuda-venv.sbatch, the `cd` and venv lines
in bench/dashboard/ride.sbatch, and the paths and `srun -p/-w` flags in serve.sh. See [fly_brain/README.md](fly_brain/README.md) and
[fly_brain/bike/README.md](fly_brain/bike/README.md) for every step.

Not committed (too large, or rebuildable): the connectome download, `brain.npz`, virtualenvs, raw
per-bin arrays (`results/bar_rates/`, `*.parquet`, `dashboard/data/*.bin`).

## Other fly-brain projects we tried

These were cloned next to our work and not modified, so they are not copied here:

| Project | Commit | What it is |
|---|---|---|
| [Lulzx/fly-brain](https://github.com/Lulzx/fly-brain) | `08cf866` | male CNS as a spiking brain in a physics-simulated body, in the browser ([live](https://lulzx.com/fly-brain/arena.html)) (MIT) |
| [solomonsealed/flybrain](https://github.com/solomonsealed/flybrain) | `06bae5d` | FlyWire brain driving flies in a 3D garden, in a Web Worker (MIT) |
| [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) | `a3db62f` | Shiu et al. FlyWire LIF model: activate / silence neurons (GPL-2.0) |

## Credits

Connectome: male CNS v1.0, HHMI Janelia FlyEM and Google Research. Simulator and Fly Bar: Spyros
([sstamou03/fly_brain](https://github.com/sstamou03/fly_brain)). Neuron model: Shiu et al.,
*Nature* 2024. Bicycle model: Meijaard et al., *Proc. R. Soc. A* 2007.
