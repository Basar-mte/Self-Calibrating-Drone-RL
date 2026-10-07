"""Training curves for one or more runs.

    python plot_training.py --runs results/sac_*_s0 --out results/figures/training_curves.png
"""
import argparse
import glob
import os

from drone_cal.utils import load_config
from drone_cal.viz import plot_training

LABELS = {"calib": "SAC + calibrator", "oracle": "SAC + true parameters", "history": "SAC + history"}


def label_of(run):
    cfg = load_config(os.path.join(run, "config.yaml"))
    mode = cfg["observation"]["mode"]
    if mode in LABELS:
        return LABELS[mode]
    return "SAC, blind" if cfg["uncertainty"]["enabled"] else "SAC, nominal training"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/figures/training_curves.png")
    args = ap.parse_args()
    runs = sorted({r for p in args.runs for r in glob.glob(p) if os.path.exists(os.path.join(r, "eval_log.csv"))})
    plot_training(runs, [label_of(r) for r in runs], args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
