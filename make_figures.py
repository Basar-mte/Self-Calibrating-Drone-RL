"""Build every figure in the README from the evaluation results.

    python make_figures.py                 # after evaluate.py, calibration_study.py and training

Reads results/eval/ (from evaluate.py) and the run folders; writes results/figures/.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from drone_cal.scenarios import SCENARIOS
from drone_cal.viz import make_gif, plot_recovery, plot_scenario_bars, plot_training, plot_trajectories_3d
from plot_training import label_of

MAIN = ["PID", "PID + calibrator", "SAC, nominal training", "SAC, blind", "SAC + history",
        "SAC + calibrator", "SAC + true parameters"]
TRIGGER = ["PID + calibrator", "PID + calibrator, no trigger", "SAC + calibrator", "SAC + calibrator, no trigger"]
TRAJ = ["PID", "SAC, blind", "SAC + calibrator", "PID + calibrator"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default="results/eval")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--gif-episode", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = pd.read_csv(os.path.join(args.eval, "summary.csv"))
    curves = dict(np.load(os.path.join(args.eval, "curves.npz")))
    labels = json.load(open(os.path.join(args.eval, "labels.json"), encoding="utf-8"))
    present = set(summary.controller)
    main = [c for c in MAIN if c in present]
    scenarios = [s for s in SCENARIOS if s in set(summary.scenario)]
    inv = {v: k for k, v in labels.items()}

    any_trace = np.load(glob.glob(os.path.join(args.eval, "traces", "*.npz"))[0])
    dt, ev = float(any_trace["dt"]), int(any_trace["event_step"])

    plot_recovery(curves, labels, scenarios, main, dt, ev, os.path.join(args.out, "recovery.png"))
    plot_scenario_bars(summary, main, scenarios, os.path.join(args.out, "scenarios.png"))
    trig = [c for c in TRIGGER if c in present]
    plot_recovery(curves, labels, ["payload", "combined", "severe"], trig, dt, ev,
                  os.path.join(args.out, "trigger_ablation.png"))

    traces = {}
    for lab in TRAJ:
        f = os.path.join(args.eval, "traces", f"{inv[lab]}__combined.npz") if lab in inv else ""
        if os.path.exists(f):
            traces[lab] = dict(np.load(f))
    plot_trajectories_3d(traces, args.gif_episode, os.path.join(args.out, "trajectories_combined.png"))

    if "SAC + calibrator" in traces:
        make_gif(traces["SAC + calibrator"], args.gif_episode, "SAC + calibrator",
                 os.path.join(args.out, "episode.gif"),
                 title="scenario: payload +45 %, one motor -35 %, wind 5 m/s, all at t = 5 s")

    runs = sorted(r for r in glob.glob("results/sac_*") if os.path.exists(os.path.join(r, "eval_log.csv")))
    plot_training(runs, [label_of(r) for r in runs], os.path.join(args.out, "training_curves.png"))
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
