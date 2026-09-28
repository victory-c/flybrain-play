# Jobs

Every Slurm job run from this repo on the OCF cluster (node `corruption`), newest last. Generated
by `python3 tools/jobs_ledger.py` from `sacct`; do not edit by hand, add context to `NOTES` there.
Commands run from the job's working directory (`.` = repo root, `fly_brain/` for most `srun` jobs).

| Job | Started | State | Time | Resources | What | Command | Logs / outputs |
|---|---|---|---|---|---|---|---|
| 36053 | 2026-09-26 21:16 | COMPLETED | 00:00:00 | 4 cpu, 8G, 1 gpu | probe: node, GPU, python and torch on the cluster | `srun -p ocf-hpc --gres=gpu:1 -c 4 --mem=8G -t 5 bash -c hostname; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv; which python3 conda; python3 --version; ls /opt/conda/bin 2>/dev/nul...` |  |
| 36054 | 2026-09-26 21:18 | COMPLETED | 00:01:07 | 32 cpu, 64G | build brain.npz from the connectome + tests.test_sim | `sbatch flybar.sbatch build` | [flybar-36054.out](logs/flybar-36054.out) |
| 36055 | 2026-09-26 21:20 | COMPLETED | 00:00:54 | 32 cpu, 64G | pilot bar: 4 drinks, 5 trials | `sbatch flybar.sbatch pilot` | [flybar-36055.out](logs/flybar-36055.out) |
| 36056 | 2026-09-26 21:22 | COMPLETED | 02:09:33 | 32 cpu, 64G | full bar + hunger + UI/web export -> results/bar.json, hunger_cocktails.json, app/public/data | `sbatch flybar.sbatch full` | [flybar-36056.out](logs/flybar-36056.out) |
| 36057 | 2026-09-26 21:27 | COMPLETED | 00:15:32 | 32 cpu, 32G | custom drink served with serve.sh | `srun -p ocf-hpc -w corruption -c 32 --mem=32G -t 20 --quiet env OMP_NUM_THREADS=32 /home/s/st/stevejobs/flybrain/venv/bin/python -m runs.serve 🧋 珍珠奶茶 --sugar 80 --caffeine 150 --ph 6.5 --trials 5` (in `./fly_brain`) |  |
| 36059 | 2026-09-26 21:43 | COMPLETED | 00:00:01 | 32 cpu, 4G | probe: CPU affinity and torch threads | `srun -p ocf-hpc -w corruption -c 32 --mem=4G -t 3 --quiet env OMP_NUM_THREADS=32 venv/bin/python -c  ⏎ import os, torch; print('affinity', len(os.sched_getaffinity(0)), 'torch threads', torch.get_num_threads(), 'OMP',...` |  |
| 36062 | 2026-09-26 23:32 | COMPLETED | 00:00:17 | 32 cpu, 32G | custom drink served with serve.sh | `srun -p ocf-hpc -w corruption -c 32 --mem=32G -t 20 --quiet env OMP_NUM_THREADS=32 /home/s/st/stevejobs/flybrain/venv/bin/python -m runs.serve 🍉 西瓜汁 --sugar 60 --ph 5.5 --trials 5` (in `./fly_brain`) |  |
| 36064 | 2026-09-27 02:13 | COMPLETED | 00:08:25 | 32 cpu, 64G | Fly Brain Live dashboard data -> dashboard/data | `sbatch dashboard.sbatch --trials 8` | [flydash-36064.out](logs/flydash-36064.out) |
| 36073 | 2026-09-27 04:54 | COMPLETED | 00:00:00 | 2 cpu, 2G, 1 gpu | probe: GPU name and driver | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 2 --mem=2G -t 3 --quiet nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` |  |
| 36074 | 2026-09-27 04:54 | FAILED | 00:00:00 | 2 cpu, 1G | probe: CUDA version | `srun -p ocf-hpc -w corruption -c 1 --mem=1G -t 2 --quiet bash -c ls /usr/local/cuda*/bin/nvcc 2>/dev/null; nvidia-smi ¦ grep -oE "CUDA Version: [0-9.]+"` |  |
| 36075 | 2026-09-27 04:55 | COMPLETED | 00:03:23 | 4 cpu, 16G, 1 gpu | build venv-cuda (torch cu121) and check the GPU | `sbatch cuda-venv.sbatch` | [cudavenv-36075.out](logs/cudavenv-36075.out) |
| 36076 | 2026-09-27 04:55 | FAILED | 00:00:00 | 8 cpu, 8G | failed: started in the repo root, where there is no brain package | `srun -p ocf-hpc -w corruption -c 8 --mem=8G -t 10 --quiet env OMP_NUM_THREADS=8 /home/s/st/stevejobs/flybrain/venv/bin/python -m tests.test_sim` |  |
| 36077 | 2026-09-27 04:55 | COMPLETED | 00:00:08 | 8 cpu, 8G | tests.test_sim on CPU after the GPU port | `srun -p ocf-hpc -w corruption -c 8 --mem=8G -t 10 --quiet --chdir=/home/s/st/stevejobs/flybrain/fly_brain env OMP_NUM_THREADS=8 /home/s/st/stevejobs/flybrain/venv/bin/python -m tests.test_sim` (in `./fly_brain`) |  |
| 36078 | 2026-09-27 04:58 | COMPLETED | 00:03:02 | 32 cpu, 64G, 1 gpu | CPU vs GPU benchmark -> results/bench_gpu.json | `sbatch bench.sbatch --devices cpu cuda --trials 1 8 32 128` | [flybench-36078.out](logs/flybench-36078.out) |
| 36079 | 2026-09-27 05:00 | FAILED | 00:00:00 | 2 cpu, 1G | probe: GPU memory (failed) | `srun -p ocf-hpc -w corruption -c 1 --mem=1G -t 2 --quiet nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv` |  |
| 36080 | 2026-09-27 05:02 | FAILED | 00:00:00 | 8 cpu, 32G, 1 gpu | failed: relative venv path | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=32G -t 15 --quiet --chdir=/home/s/st/stevejobs/flybrain/fly_brain env OMP_NUM_THREADS=8 venv-cuda/bin/python -m runs.bench_gpu --devices cuda --trials 512 --out re...` (in `./fly_brain`) | `results/bench_gpu_512.json` |
| 36081 | 2026-09-27 05:02 | FAILED | 00:00:00 | 8 cpu, 32G, 1 gpu | failed: relative venv path | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=32G -t 15 --chdir=/home/s/st/stevejobs/flybrain/fly_brain env OMP_NUM_THREADS=8 venv-cuda/bin/python -m runs.bench_gpu --devices cuda --trials 512 --out results/be...` (in `./fly_brain`) | `results/bench_gpu_512.json` |
| 36082 | 2026-09-27 05:02 | COMPLETED | 00:02:04 | 8 cpu, 32G, 1 gpu | 512-trial GPU batch -> results/bench_gpu_512.json | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=32G -t 15 --chdir=/home/s/st/stevejobs/flybrain/fly_brain env OMP_NUM_THREADS=8 /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.bench_gpu --devices cuda...` (in `./fly_brain`) | `results/bench_gpu_512.json` |
| 36083 | 2026-09-27 05:11 | COMPLETED | 00:00:07 | 4 cpu, 8G | bike + closed-loop sanity tests | `srun -p ocf-hpc -w corruption -c 4 --mem=8G -t 10 --quiet ../venv/bin/python -m tests.test_ride` (in `./fly_brain`) |  |
| 36084 | 2026-09-27 05:11 | COMPLETED | 00:00:02 | 4 cpu, 8G | interface smoke test: senses, readout, decoder on a toy peloton | `srun -p ocf-hpc -w corruption -c 4 --mem=8G -t 10 --quiet ../venv/bin/python -` (in `./fly_brain`) | [inline script](logs/inline/36084.py) |
| 36085 | 2026-09-27 05:12 | COMPLETED | 00:00:37 | 8 cpu, 40G, 1 gpu | open loop: brain in the loop, prior decoder, no steering | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet ../venv-cuda/bin/python -m runs.ride --open-loop --riders 8 --seconds 3 --trace results/ride_pilot_trace.json` (in `./fly_brain`) | `results/ride_pilot_trace.json` |
| 36086 | 2026-09-27 05:13 | COMPLETED | 00:00:06 | 4 cpu, 8G | PD rider without a brain (oracle) | `srun -p ocf-hpc -w corruption -c 4 --mem=8G -t 10 --quiet ../venv/bin/python -m runs.ride --oracle --riders 32 --seconds 10 --trace results/ride_oracle_trace.json` (in `./fly_brain`) | `results/ride_oracle_trace.json` |
| 36087 | 2026-09-27 05:15 | COMPLETED | 00:01:09 | 8 cpu, 40G, 1 gpu | passenger screen, pilot | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 20 --quiet ../venv-cuda/bin/python -m runs.screen --riders 8 --seconds 6` (in `./fly_brain`) |  |
| 36088 | 2026-09-27 05:18 | COMPLETED | 00:01:05 | 8 cpu, 40G, 1 gpu | which sensory population saturates the brain (odour goal diagnosis) | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet ../venv-cuda/bin/python -` (in `./fly_brain`) | [inline script](logs/inline/36088.py) |
| 36089 | 2026-09-27 05:19 | COMPLETED | 00:01:43 | 8 cpu, 40G, 1 gpu | passenger screen -> results/screen.json, screen_samples.npz | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 25 --quiet ../venv-cuda/bin/python -m runs.screen --riders 16 --seconds 8` (in `./fly_brain`) |  |
| 36090 | 2026-09-27 05:22 | FAILED | 00:00:03 | 8 cpu, 48G, 1 gpu | failed: wrong path to screen_samples.npz | `sbatch ride.sbatch --init fly_brain/results/screen_samples.npz --riders 48 --generations 15 --seconds 10 --out results/ride.json` | `results/ride.json`, [flyride-36090.err](logs/flyride-36090.err), [flyride-36090.out](logs/flyride-36090.out) |
| 36091 | 2026-09-27 05:23 | COMPLETED | 00:50:32 | 8 cpu, 48G, 1 gpu | CEM learning, sigma0 0.3 -> results/ride.json | `sbatch ride.sbatch --init results/screen_samples.npz --riders 48 --generations 15 --seconds 10 --out results/ride.json` | `results/ride.json`, [flyride-36091.out](logs/flyride-36091.out) |
| 36092 | 2026-09-27 06:13 | COMPLETED | 00:51:17 | 8 cpu, 48G, 1 gpu | CEM learning, sigma0 0.1 -> results/ride_b.json (the best decoder) | `sbatch ride.sbatch --init results/screen_samples.npz --riders 48 --generations 15 --seconds 10 --sigma0 0.1 --seed 1 --out results/ride_b.json` | `results/ride_b.json`, [flyride-36092.out](logs/flyride-36092.out) |
| 36093 | None | CANCELLED | 00:00:00 | 1 gpu | cancelled, rerun on CPU as 36094 | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 25 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --replay results/ride.json --riders 16 --seconds 12 --seed 7 --trace results/ride...` (in `./fly_brain`) | `results/ride_trace.json` |
| 36094 | 2026-09-27 06:25 | COMPLETED | 00:08:26 | 32 cpu, 48G | replay of ride.json -> results/ride_trace.json | `srun -p ocf-hpc -w corruption -c 32 --mem=48G -t 40 --quiet env OMP_NUM_THREADS=32 /home/s/st/stevejobs/flybrain/venv/bin/python -m runs.ride --replay results/ride.json --riders 16 --seconds 12 --seed 7 --device cpu -...` (in `./fly_brain`) | `results/ride_trace.json` |
| 36095 | 2026-09-27 07:05 | COMPLETED | 00:03:26 | 8 cpu, 40G, 1 gpu | replay of the best decoder -> results/ride_b_trace.json (the /ride/ demo) | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --replay results/ride_b.json --riders 16 --seconds 15 --seed 7 --trace results/ri...` (in `./fly_brain`) | `results/ride_b_trace.json` |
| 36101 | None | CANCELLED | 00:00:00 | 1 gpu | cancelled: is a weak odour goal stable? | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -` (in `./fly_brain`) | [inline script](logs/inline/36101.py) |
| 36102 | None | CANCELLED | 00:00:00 |  | cancelled: HS lane cue check | `srun -p ocf-hpc -w corruption -c 2 --mem=6G -t 5 --quiet /home/s/st/stevejobs/flybrain/venv/bin/python -` (in `./fly_brain`) | [inline script](logs/inline/36102.py) |
| 36103 | 2026-09-28 08:15 | COMPLETED | 00:56:28 | 8 cpu, 48G, 1 gpu | lane keeping: screen with HS lane cue -> CEM -> replay | `sbatch ride-lane.sbatch` | [flylane-36103.out](logs/flylane-36103.out) |
| 36104 | None | CANCELLED | 00:00:00 |  |  | `srun -p ocf-hpc -w corruption -c 1 --mem=256M -t 1 --quiet bash -c timeout 10 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ~/.ssh/flybrain_play_deploy -o IdentitiesOnly=yes -T git@github.com 2>&1 ¦ head -1` |  |
| 36105 | 2026-09-28 08:52 | FAILED | 00:00:03 | 8 cpu, 40G, 1 gpu |  | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --init results/screen_lane_samples.npz --lane hs --dagger 2 --riders 4 --seconds ...` (in `./fly_brain`) | `/tmp/claude-93015/-home-s-st-stevejobs-flybrain/ff3d4c1f-4087-4f88-b1bd-f60320256b54/scratchpad/dagger_smoke.json` |
| 36106 | 2026-09-28 08:52 | COMPLETED | 00:00:27 | 8 cpu, 40G, 1 gpu |  | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --init results/screen_lane_samples.npz --lane hs --dagger 2 --riders 4 --seconds ...` (in `./fly_brain`) | `results/dagger_smoke.json` |
| 36107 | 2026-09-28 08:53 | COMPLETED | 00:35:11 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-dagger.sbatch` | [flydagger-36107.out](logs/flydagger-36107.out) |
| 36108 | 2026-09-28 09:12 | COMPLETED | 00:54:16 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-offroad.sbatch results/ride_lane.json results/ride_road.json` | [flyroad-36108.out](logs/flyroad-36108.out) |
| 36109 | 2026-09-28 09:27 | COMPLETED | 00:51:47 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-offroad2.sbatch results/ride_lane.json results/ride_road2.json` | [flyroad2-36109.out](logs/flyroad2-36109.out) |
| 36110 | 2026-09-28 09:33 | COMPLETED | 00:05:10 | 48 cpu, 64G |  | `srun -p ocf-hpc -w corruption -c 48 --mem=64G -t 60 --quiet env OMP_NUM_THREADS=48 /home/s/st/stevejobs/flybrain/venv/bin/python -m runs.screen --riders 16 --seconds 8 --lane hs --device cpu --out results/screen_lane_...` (in `./fly_brain`) | `results/screen_lane_psi.json` |
| 36111 | 2026-09-28 10:08 | COMPLETED | 00:01:41 | 8 cpu, 40G, 1 gpu |  | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.screen --riders 16 --seconds 8 --lane hs --polarity physio --out results/screen_physio...` (in `./fly_brain`) | `results/screen_physio.json` |
| 36112 | 2026-09-28 10:11 | COMPLETED | 00:00:45 | 8 cpu, 40G, 1 gpu |  | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.probe --pop HS VS HALT --extra 30 --brains 16 --ms 1000 --out results/probe.json` (in `./fly_brain`) | `results/probe.json` |
| 36113 | 2026-09-28 10:13 | CANCELLED | 00:04:13 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-lanedn.sbatch  results/ride_lanedn.json` | [flylanedn-36113.err](logs/flylanedn-36113.err), [flylanedn-36113.out](logs/flylanedn-36113.out) |
| 36114 | 2026-09-28 10:13 | CANCELLED | 00:04:13 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-lanedn.sbatch hs_heading=150,hs_lane=50 results/ride_lanedn_strong.json` | [flylanedn-36114.err](logs/flylanedn-36114.err), [flylanedn-36114.out](logs/flylanedn-36114.out) |
| 36115 | 2026-09-28 10:18 | CANCELLED | 00:04:56 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-lanedn.sbatch  results/ride_lanedn.json` | [flylanedn-36115.err](logs/flylanedn-36115.err), [flylanedn-36115.out](logs/flylanedn-36115.out) |
| 36116 | 2026-09-28 10:18 | CANCELLED | 00:04:56 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-lanedn.sbatch hs_heading=150,hs_lane=50 results/ride_lanedn_strong.json` | [flylanedn-36116.err](logs/flylanedn-36116.err), [flylanedn-36116.out](logs/flylanedn-36116.out) |
| 36117 | 2026-09-28 10:22 | CANCELLED | 00:31:51 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-lanedn.sbatch  results/ride_lanedn_narrow.json 0.25 8` | [flylanedn-36117.err](logs/flylanedn-36117.err), [flylanedn-36117.out](logs/flylanedn-36117.out) |
| 36118 | 2026-09-28 10:23 | CANCELLED | 00:18:53 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-lanedn.sbatch hs_heading=150,hs_lane=50 results/ride_lanedn_strong.json 0.25 9` | [flylanedn-36118.err](logs/flylanedn-36118.err), [flylanedn-36118.out](logs/flylanedn-36118.out) |
| 36119 | 2026-09-28 10:23 | CANCELLED | 00:02:19 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-lanedn.sbatch  results/ride_lanedn_tiny.json 0.1 10` | [flylanedn-36119.err](logs/flylanedn-36119.err), [flylanedn-36119.out](logs/flylanedn-36119.out) |
| 36120 | 2026-09-28 10:25 | CANCELLED | 00:08:06 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-lanefilter.sbatch` | [flyfilter-36120.err](logs/flyfilter-36120.err), [flyfilter-36120.out](logs/flyfilter-36120.out) |
| 36121 | 2026-09-28 10:33 | CANCELLED | 00:08:22 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-fast.sbatch hs_heading=150,hs_lane=50 results/ride_fast.json 0.25 12` | [flyfast-36121.err](logs/flyfast-36121.err), [flyfast-36121.out](logs/flyfast-36121.out) |
| 36122 | 2026-09-28 10:40 | COMPLETED | 00:00:15 | 4 cpu, 8G |  | `srun -p ocf-hpc -w corruption -c 4 --mem=8G -t 5 --quiet /home/s/st/stevejobs/flybrain/venv/bin/python -` (in `./fly_brain`) |  |
| 36123 | 2026-09-28 10:41 | COMPLETED | 00:00:19 | 4 cpu, 8G |  | `srun -p ocf-hpc -w corruption -c 4 --mem=8G -t 5 --quiet /home/s/st/stevejobs/flybrain/venv/bin/python -` (in `./fly_brain`) |  |
| 36124 | 2026-09-28 10:42 | COMPLETED | 01:06:33 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-easy.sbatch results/ride_lane.json results/ride_easy_seeded.json 0.08 0.25 21` | [flyeasy-36124.out](logs/flyeasy-36124.out) |
| 36125 | 2026-09-28 10:42 | CANCELLED | 00:02:55 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-easy.sbatch results/zero_theta.json results/ride_easy_zero.json 0.15 0.15 22` | [flyeasy-36125.err](logs/flyeasy-36125.err), [flyeasy-36125.out](logs/flyeasy-36125.out) |
| 36126 | 2026-09-28 10:44 | COMPLETED | 01:22:42 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-easy-c.sbatch results/zero_theta.json results/ride_easy_zero.json 0.15 0.15 22` | [flyeasyc-36126.out](logs/flyeasyc-36126.out) |
| 36127 | 2026-09-28 10:54 | COMPLETED | 01:24:01 | 8 cpu, 48G, 1 gpu |  | `sbatch ride-slow.sbatch results/ride_lane.json results/ride_slow.json 0.08 0.25 23` | [flyslow-36127.out](logs/flyslow-36127.out) |
| 36128 | 2026-09-28 11:50 | COMPLETED | 00:25:32 | 8 cpu, 48G, 1 gpu |  | `sbatch sweep-gain.sbatch handsoff gain_+0.00 gain_-0.29 gain_-1.00 gain_-2.00 gain_-4.00 gain_-0.50 gain_+1.00` | [flysweep-36128.out](logs/flysweep-36128.out) |
| 36129 | 2026-09-28 12:12 | COMPLETED | 00:04:35 | 8 cpu, 40G, 1 gpu |  | `srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 30 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --replay results/ride_slow_mu.json --tau-lane-ms 300 --lane-filter results/probe....` (in `./fly_brain`) | `results/ride_slow_mu_trace.json` |

## Full command lines

**36053** (bash)

```bash
srun -p ocf-hpc --gres=gpu:1 -c 4 --mem=8G -t 5 bash -c hostname; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv; which python3 conda; python3 --version; ls /opt/conda/bin 2>/dev/null | head -3; python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())" 2>&1; df -h /scratch /tmp 2>/dev/null | tail -2
```

**36059** (env)

```bash
srun -p ocf-hpc -w corruption -c 32 --mem=4G -t 3 --quiet env OMP_NUM_THREADS=32 venv/bin/python -c 
import os, torch; print('affinity', len(os.sched_getaffinity(0)), 'torch threads', torch.get_num_threads(), 'OMP', os.environ.get('OMP_NUM_THREADS'))
```

**36080** (env)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=32G -t 15 --quiet --chdir=/home/s/st/stevejobs/flybrain/fly_brain env OMP_NUM_THREADS=8 venv-cuda/bin/python -m runs.bench_gpu --devices cuda --trials 512 --out results/bench_gpu_512.json
```

**36081** (env)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=32G -t 15 --chdir=/home/s/st/stevejobs/flybrain/fly_brain env OMP_NUM_THREADS=8 venv-cuda/bin/python -m runs.bench_gpu --devices cuda --trials 512 --out results/bench_gpu_512.json
```

**36082** (env)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=32G -t 15 --chdir=/home/s/st/stevejobs/flybrain/fly_brain env OMP_NUM_THREADS=8 /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.bench_gpu --devices cuda --trials 512 --out results/bench_gpu_512.json
```

**36093** (python)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 25 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --replay results/ride.json --riders 16 --seconds 12 --seed 7 --trace results/ride_trace.json
```

**36094** (env)

```bash
srun -p ocf-hpc -w corruption -c 32 --mem=48G -t 40 --quiet env OMP_NUM_THREADS=32 /home/s/st/stevejobs/flybrain/venv/bin/python -m runs.ride --replay results/ride.json --riders 16 --seconds 12 --seed 7 --device cpu --trace results/ride_trace.json
```

**36095** (python)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --replay results/ride_b.json --riders 16 --seconds 15 --seed 7 --trace results/ride_b_trace.json
```

**36105** (python)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --init results/screen_lane_samples.npz --lane hs --dagger 2 --riders 4 --seconds 0.5 --out /tmp/claude-93015/-home-s-st-stevejobs-flybrain/ff3d4c1f-4087-4f88-b1bd-f60320256b54/scratchpad/dagger_smoke.json
```

**36106** (python)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --init results/screen_lane_samples.npz --lane hs --dagger 2 --riders 4 --seconds 0.5 --out results/dagger_smoke.json
```

**36110** (env)

```bash
srun -p ocf-hpc -w corruption -c 48 --mem=64G -t 60 --quiet env OMP_NUM_THREADS=48 /home/s/st/stevejobs/flybrain/venv/bin/python -m runs.screen --riders 16 --seconds 8 --lane hs --device cpu --out results/screen_lane_psi.json
```

**36111** (python)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 15 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.screen --riders 16 --seconds 8 --lane hs --polarity physio --out results/screen_physio.json
```

**36129** (python)

```bash
srun -p ocf-hpc -w corruption --gres=gpu:1 -c 8 --mem=40G -t 30 --quiet /home/s/st/stevejobs/flybrain/venv-cuda/bin/python -m runs.ride --replay results/ride_slow_mu.json --tau-lane-ms 300 --lane-filter results/probe.json --readout lane --lane hs --gains hs_heading=150,hs_lane=50 --v0 5.5 --gust 5 --riders 16 --seconds 20 --seed 7 --trace results/ride_slow_mu_trace.json
```
