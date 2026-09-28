# flybrain-play

A fruit fly's whole central nervous system, 166,700 neurons and every synapse between them, run as
a spiking network. We give it drinks, record every neuron, and put it in charge of a road bike.

**Live demos: https://flybrain-play.vercel.app**

| Demo | What you see |
|---|---|
| [The fly rides a Tarmac SL9](https://flybrain-play.vercel.app/ride/) | 3D replay: the brain balances a Specialized S-Works Tarmac SL9 in side gusts. Chase cam, side cam, or the fly's own eyes. Also: [charts](https://flybrain-play.vercel.app/ride/charts), [PD rider without a brain](https://flybrain-play.vercel.app/ride/oracle), [brain in the loop but not steering](https://flybrain-play.vercel.app/ride/open-loop) |
| [Fly Brain Live](https://flybrain-play.vercel.app/dashboard/) | Whole-brain activity in 50 ms frames while the fly tastes 12 drinks (orange juice, bubble tea, cola, espresso, lager, tsipouro, ...), with the feeding motor neuron MN9 |
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
   `pack_dashboard.py`): time-resolved replays of the whole 3D neuron cloud (140,638 points) for 12 drinks.
3. **GPU port** (`fly_brain/brain/sim.py`, `device="cuda"`; `runs/bench_gpu.py`). A single trial is
   Python-loop bound (~8.5 s per simulated second either way); batching wins: 128 trials in 34.6 s on
   one RTX A6000.
4. **The fly rides a bike** (`fly_brain/bike/`, `brain/loop.py`, `runs/screen.py`, `runs/ride.py`,
   `ride.sbatch`). Details in [fly_brain/bike/README.md](fly_brain/bike/README.md). In short:
   - bike state is turned into firing of real balance, wind, optic-flow and leg sensory neurons;
     descending-neuron firing is decoded into steering torque, pedalling and braking;
   - the walking command neurons of the literature stay silent, but flight-steering descending
     neurons (DNp20, DNp22, DNg46, ...) track roll rate with opposite sign left and right, so the fly
     steers the bike with its flight-stabilisation reflex;
   - a 29-weight decoder fitted to a PD teacher (R² 0.75) and refined with the cross-entropy method
     keeps 16 riders up for a mean of 14.1 s out of 15 s; 10 of 16 never fall. The connectome is
     never changed.
5. **Demo site** (`site/`, `vercel.json`): `site/build.sh` builds the app and collects the pages.

```
flybrain-play/
├── fly_brain/        the simulator + our additions (brain/, runs/, bike/, export/, dashboard/, app/, results/)
├── site/             landing page and build script for the Vercel site
├── *.sbatch          Slurm jobs: flybar (CPU), cuda-venv, bench, dashboard, ride (GPU)
├── serve.sh          serve one drink via srun
└── logs/             Slurm output of the runs whose results are committed
```

## Running it

The web demos need nothing but a browser. To rebuild the site locally (Node 20+):

```bash
bash site/build.sh && npx serve site/dist
```

To run the simulations you need Python 3.11+ with `numpy pandas pyarrow torch` and the male CNS v1.0
files (~1.1 GB, [male-cns.janelia.org/download](https://male-cns.janelia.org/download)) in
`fly_brain/data/`; `python -m brain.build_brain` turns them into `brain.npz`. The sbatch scripts are
written for the OCF `corruption` node (`/home/s/st/stevejobs/flybrain` paths, partition `ocf-hpc`); edit
`ROOT` and the `#SBATCH` lines for another cluster. See [fly_brain/README.md](fly_brain/README.md) and
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
