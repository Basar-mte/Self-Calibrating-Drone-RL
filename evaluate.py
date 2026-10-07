"""Fly every controller through every test scenario and compare them.

    python evaluate.py --runs results/sac_calib_s0 results/sac_blind_s0 --pid
    python evaluate.py --runs results/sac_*_s* --pid --episodes 100     # all seeds

All controllers meet exactly the same drones: scenario k always uses seed 50_000 + k, so
the starts, goals, which motor fails and the wind direction are identical for every row.
Runs whose config gives the same controller (for example the same recipe with three
seeds) are grouped; the tables report the mean over seeds and the spread between seeds.

Writes to --out (default results/eval):
    summary.csv / summary.json   one row per controller and scenario
    curves.npz                   distance-to-goal over time (median and quartiles)
    traces/<controller>__<scenario>.npz   full recordings of the first episodes, for figures
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from drone_cal.env import DroneEnv
from drone_cal.evaluation import episode_metrics, run_episodes, summarise
from drone_cal.pid import CascadedPID
from drone_cal.sac import SACAgent, SACController
from drone_cal.scenarios import SCENARIOS, EvalScenario
from drone_cal.utils import deep_copy, get_device, load_config

SEED_BASE = 50_000
TRACE_EPISODES = 8


def controller_label(cfg, trigger=True):
    mode = cfg["observation"]["mode"]
    if mode == "calib":
        return "SAC + calibrator" if trigger else "SAC + calibrator, no trigger"
    if mode == "oracle":
        return "SAC + true parameters"
    if mode == "history":
        return "SAC + history"
    return "SAC, blind" if cfg["uncertainty"]["enabled"] else "SAC, nominal training"


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def load_sac(run_dir, device):
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    probe = DroneEnv(cfg, 1)
    agent = SACAgent(probe.obs_dim, probe.action_dim, cfg["sac"], device)
    ck = agent.load(os.path.join(run_dir, "best.pt"))
    return cfg, agent, ck.get("step")


def evaluate_controller(make_controller, cfg, obs_mode, trigger, scenarios, episodes, keep_traces):
    """Returns {scenario: (metrics, rec)} for one controller."""
    out = {}
    for k, name in enumerate(scenarios):
        sc = EvalScenario(name, cfg)
        env = DroneEnv(cfg, episodes, seed=SEED_BASE + k, sampler=sc, auto_reset=False,
                       obs_mode=obs_mode, trigger=trigger)
        ctl = make_controller(episodes)
        rec = run_episodes(env, ctl, record=keep_traces)
        m = episode_metrics(rec["dist"], rec["crash_step"], sc.event_step, env.dt)
        out[name] = (m, rec, sc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="training run folders (globs allowed)")
    ap.add_argument("--pid", action="store_true", help="also evaluate the PID with and without the calibrator")
    ap.add_argument("--no-trigger-ablation", action="store_true",
                    help="skip re-flying calibrator agents with the change detector switched off")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--config", default="configs/default.yaml", help="physics config for the PID")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    os.makedirs(os.path.join(args.out, "traces"), exist_ok=True)
    runs = sorted({r for pat in args.runs for r in glob.glob(pat) if os.path.isfile(os.path.join(r, "best.pt"))})

    jobs = []   # (label, source, make_controller, cfg, obs_mode, trigger)
    for run in runs:
        cfg, agent, step = load_sac(run, device)
        label = controller_label(cfg)
        jobs.append((label, run, lambda n, a=agent: SACController(a), cfg, cfg["observation"]["mode"], True))
        if cfg["observation"]["mode"] == "calib" and not args.no_trigger_ablation:
            jobs.append((controller_label(cfg, trigger=False), run,
                         lambda n, a=agent: SACController(a), cfg, "calib", False))
    if args.pid:
        cfg = load_config(args.config)
        jobs.append(("PID", "classical", lambda n, c=cfg: CascadedPID(c, n, False), cfg, "base", True))
        jobs.append(("PID + calibrator", "classical", lambda n, c=cfg: CascadedPID(c, n, True), cfg, "base", True))
        jobs.append(("PID + calibrator, no trigger", "classical",
                     lambda n, c=cfg: CascadedPID(c, n, True), cfg, "base", False))

    per_seed = []       # one summary row per (label, source, scenario)
    pooled = {}         # (label, scenario) -> list of metric dicts
    curves = {}
    traced = set()
    for label, source, make, cfg, mode, trig in jobs:
        print(f"{label:32s} {source}")
        keep = (label, ) not in traced
        res = evaluate_controller(make, cfg, mode, trig, args.scenarios, args.episodes, keep)
        for name, (m, rec, sc) in res.items():
            row = {"controller": label, "source": source, "scenario": name, **summarise(m)}
            per_seed.append(row)
            pooled.setdefault((label, name), []).append((m, rec["dist"]))
            if keep:
                np.savez_compressed(
                    os.path.join(args.out, "traces", f"{slug(label)}__{name}.npz"),
                    event_step=sc.event_step, dt=cfg["sim"]["dt"],
                    **{k: (v[:, :TRACE_EPISODES] if isinstance(v, np.ndarray) and v.ndim >= 2 and k != "goal"
                           else v[:TRACE_EPISODES]) for k, v in rec.items()})
            print(f"    {name:12s} crash {row['crash_rate']:5.2f}  rmse {row['rmse_post']:6.3f}  "
                  f"peak {row['peak']:6.3f}  settle {row['settle_median']:5.2f} s  hold {row['hold']:6.3f}")
        traced.add((label, ))

    # summary over seeds: mean of per-seed summaries, plus the spread between seeds
    df = pd.DataFrame(per_seed)
    df.to_csv(os.path.join(args.out, "per_seed.csv"), index=False)
    metrics = ["crash_rate", "success_rate", "pre", "peak", "rmse_post", "settle_median", "hold", "settled_rate"]
    agg = df.groupby(["controller", "scenario"], sort=False)[metrics].agg(["mean", "std", "count"])
    rows = []
    for (label, name), g in agg.iterrows():
        r = {"controller": label, "scenario": name, "seeds": int(g[("crash_rate", "count")])}
        for mtr in metrics:
            r[mtr] = g[(mtr, "mean")]
            r[mtr + "_sd"] = g[(mtr, "std")] if r["seeds"] > 1 else float("nan")
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False, float_format="%.4f")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes_per_seed": args.episodes, "seed_base": SEED_BASE,
                   "scenarios": {s: SCENARIOS[s]["text"] for s in args.scenarios},
                   "rows": rows}, f, indent=2, default=float)

    # distance-over-time curves, pooled over seeds, non-crashed episodes only
    for (label, name), lst in pooled.items():
        D = np.concatenate([d for _, d in lst], axis=1)
        crashed = np.concatenate([m["crashed"] for m, _ in lst])
        ok = D[:, ~crashed]
        key = f"{slug(label)}__{name}"
        if ok.shape[1]:
            curves[key + "__q"] = np.nanpercentile(ok, [25, 50, 75], axis=1)
        curves[key + "__crashed"] = np.array([crashed.mean()])
    np.savez_compressed(os.path.join(args.out, "curves.npz"), **curves)
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({slug(l): l for l, *_ in jobs}, f, indent=2)
    print(f"\nwrote {args.out}/summary.csv")
    pd.set_option("display.width", 200)
    print(summary[["controller", "scenario", "seeds", "crash_rate", "rmse_post", "peak",
                   "settle_median", "hold"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
