"""Write JOBS.md: every Slurm job run from this repo, with its exact command line, from sacct.

usage: python3 tools/jobs_ledger.py [--since 2026-09-26]

Slurm keeps each job's submit line (sbatch ... / srun ...), so srun jobs that left no log file are
recorded too. Jobs that fed Python on stdin (`python -`) have their script in logs/inline/<jobid>.py
when it could be recovered. NOTES below adds one line of context where the command does not say it.
"""
import argparse
import getpass
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["JobID", "JobName", "State", "Start", "Elapsed", "AllocTRES", "WorkDir", "SubmitLine"]

NOTES = {
    "36053": "probe: node, GPU, python and torch on the cluster",
    "36054": "build brain.npz from the connectome + tests.test_sim",
    "36055": "pilot bar: 4 drinks, 5 trials",
    "36056": "full bar + hunger + UI/web export -> results/bar.json, hunger_cocktails.json, app/public/data",
    "36057": "custom drink served with serve.sh",
    "36059": "probe: CPU affinity and torch threads",
    "36062": "custom drink served with serve.sh",
    "36064": "Fly Brain Live dashboard data -> dashboard/data",
    "36073": "probe: GPU name and driver",
    "36074": "probe: CUDA version",
    "36075": "build venv-cuda (torch cu121) and check the GPU",
    "36076": "failed: started in the repo root, where there is no brain package",
    "36077": "tests.test_sim on CPU after the GPU port",
    "36078": "CPU vs GPU benchmark -> results/bench_gpu.json",
    "36079": "probe: GPU memory (failed)",
    "36080": "failed: relative venv path",
    "36081": "failed: relative venv path",
    "36082": "512-trial GPU batch -> results/bench_gpu_512.json",
    "36083": "bike + closed-loop sanity tests",
    "36084": "interface smoke test: senses, readout, decoder on a toy peloton",
    "36085": "open loop: brain in the loop, prior decoder, no steering",
    "36086": "PD rider without a brain (oracle)",
    "36087": "passenger screen, pilot",
    "36088": "which sensory population saturates the brain (odour goal diagnosis)",
    "36089": "passenger screen -> results/screen.json, screen_samples.npz",
    "36090": "failed: wrong path to screen_samples.npz",
    "36091": "CEM learning, sigma0 0.3 -> results/ride.json",
    "36092": "CEM learning, sigma0 0.1 -> results/ride_b.json (the best decoder)",
    "36093": "cancelled, rerun on CPU as 36094",
    "36094": "replay of ride.json -> results/ride_trace.json",
    "36095": "replay of the best decoder -> results/ride_b_trace.json (the /ride/ demo)",
    "36101": "cancelled: is a weak odour goal stable?",
    "36102": "cancelled: HS lane cue check",
    "36103": "lane keeping: screen with HS lane cue -> CEM -> replay",
    "36487": "hands-off on the road (5.5 m/s, 5 Nm), GPU copy queued while the GPUs were busy",
    "36488": "PD rider on the road (5.5 m/s, 5 Nm) -> results/ride_oracle_trace.json (/ride/oracle)",
    "36490": "hands-off on the road (5.5 m/s, 5 Nm), CPU -> results/ride_pilot_road_trace.json (/ride/open-loop)",
    "36492": "replay on the road with the whole brain recorded, CPU -> results/ride_road_trace.json + ride_road_brain.npz (/ride/)",
    "36493": "lane keeping round 2 (lane-next.sbatch): batch centring + look-ahead/gain replays, operating-point CEM, held-out seed 59 -> results/lane/",
    "36494": "/ride/ page replays with the brain recorded: round-2 decoder and hands-off (ride-page.sbatch)",
}


def sacct(since):
    out = subprocess.run(
        ["sacct", "-u", getpass.getuser(), "-S", since, "-X", "--parsable2", "--format=" + ",".join(FIELDS)],
        capture_output=True, text=True, check=True).stdout
    # a submit line may span several lines (python -c "..."); a record starts with "<jobid>|"
    records, cur = [], None
    for line in out.splitlines()[1:]:
        if re.match(r"^\d+(_\d+)?\|", line):
            if cur is not None:
                records.append(cur)
            cur = line
        elif cur is not None:
            cur += "\n" + line
    if cur is not None:
        records.append(cur)
    jobs = []
    for r in records:
        parts = r.split("|", len(FIELDS) - 1)
        jobs.append(dict(zip(FIELDS, parts)))
    return jobs


def uses_gpu(job):
    """AllocTRES here has no gres/gpu, so read it from the srun flags or the sbatch script."""
    cmd = job["SubmitLine"]
    if "--gres=gpu" in cmd:
        return True
    m = re.match(r"sbatch\s+(\S+\.sbatch)", cmd)
    script = Path(job["WorkDir"]) / m.group(1) if m else None
    return bool(script and script.exists() and re.search(r"#SBATCH\s+--gres=gpu", script.read_text()))


def resources(job):
    t = dict(kv.split("=", 1) for kv in job["AllocTRES"].split(",") if "=" in kv)
    s = f"{t.get('cpu', '?')} cpu, {t.get('mem', '?')}" if t else ""
    return ", ".join(x for x in (s, "1 gpu" if uses_gpu(job) else "") if x)


def outputs(job):
    cmd, jid, name = job["SubmitLine"], job["JobID"], job["JobName"]
    found = [f"`{m}`" for m in re.findall(r"--(?:out|trace)\s+(\S+)", cmd)]
    found += [f"[{p.name}](logs/{p.name})" for p in sorted((ROOT / "logs").glob(f"*-{jid}.*")) if p.stat().st_size]
    inline = ROOT / "logs" / "inline" / f"{jid}.py"
    if inline.exists():
        found.append(f"[inline script](logs/inline/{jid}.py)")
    return ", ".join(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-26")
    a = ap.parse_args()
    jobs = [j for j in sacct(a.since) if j["WorkDir"].startswith(str(ROOT))]
    lines = [
        "# Jobs",
        "",
        "Every Slurm job run from this repo on the OCF cluster (node `corruption`), newest last. Generated",
        "by `python3 tools/jobs_ledger.py` from `sacct`; do not edit by hand, add context to `NOTES` there.",
        "Commands run from the job's working directory (`.` = repo root, `fly_brain/` for most `srun` jobs).",
        "",
        "| Job | Started | State | Time | Resources | What | Command | Logs / outputs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for j in jobs:
        cmd = j["SubmitLine"].strip()
        short = cmd.replace("\n", " ⏎ ")
        short = re.sub(r"(srun|sbatch)\s", r"\1 ", short)
        if len(short) > 220:
            short = short[:217] + "..."
        wd = j["WorkDir"].replace(str(ROOT), ".") or "."
        cell = f"`{short.replace('|', '¦')}`" + (f" (in `{wd}`)" if wd != "." else "")
        lines.append(f"| {j['JobID']} | {j['Start'].replace('T', ' ')[:16]} | {j['State'].split(' ')[0]} | {j['Elapsed']} "
                     f"| {resources(j)} | {NOTES.get(j['JobID'], '')} | {cell} | {outputs(j)} |")
    long = [j for j in jobs if len(j["SubmitLine"]) > 220 or "\n" in j["SubmitLine"]]
    if long:
        lines += ["", "## Full command lines", ""]
        for j in long:
            lines += [f"**{j['JobID']}** ({j['JobName']})", "", "```bash", j["SubmitLine"].strip(), "```", ""]
    (ROOT / "JOBS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"JOBS.md: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
