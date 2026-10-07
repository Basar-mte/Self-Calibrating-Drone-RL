"""Watch a controller cope when uncertainty arrives mid-flight.

    python demo.py                                   # pretrained self-calibrating agent, "combined" fault
    python demo.py --scenario severe
    python demo.py --run results/sac_blind_s0        # the same flight without calibration
    python demo.py --pid                             # the classical controller with the calibrator

Prints one line per flight and saves, in results/demo/ (or --out):
    trajectory.png   the first flight in 3D, before and after the change
    episode.gif      the first flight, animated next to the calibrator's estimates (unless --no-gif)

The control loop is written out below in plain form: observe, act, step.
"""
import argparse
import os

import numpy as np
import torch

from drone_cal.env import DroneEnv
from drone_cal.pid import CascadedPID
from drone_cal.sac import SACAgent, SACController
from drone_cal.scenarios import SCENARIOS, EvalScenario
from drone_cal.utils import load_config
from drone_cal.viz import make_gif, plot_trajectories_3d

DEFAULT_RUN = "results/sac_calib_s0"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN, help="a training run folder with best.pt")
    ap.add_argument("--pid", action="store_true", help="fly the PID + calibrator instead of an agent")
    ap.add_argument("--scenario", default="combined", choices=list(SCENARIOS))
    ap.add_argument("--flights", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    if args.pid:
        cfg = load_config("configs/default.yaml")
        controller = CascadedPID(cfg, args.flights, use_calibration=True)
        label = "PID + calibrator"
    else:
        cfg = load_config(os.path.join(args.run, "config.yaml"))
        probe = DroneEnv(cfg, 1)
        agent = SACAgent(probe.obs_dim, probe.action_dim, cfg["sac"], torch.device("cpu"))
        agent.load(os.path.join(args.run, "best.pt"))
        controller = SACController(agent)
        mode = cfg["observation"]["mode"]
        label = {"calib": "SAC + calibrator", "base": "SAC, blind", "history": "SAC + history",
                 "oracle": "SAC + true parameters"}[mode]

    scenario = EvalScenario(args.scenario, cfg)
    env = DroneEnv(cfg, args.flights, seed=args.seed, sampler=scenario, auto_reset=False)
    print(f"{label}: {args.flights} flights, scenario '{args.scenario}' ({scenario.text}) at t = "
          f"{scenario.event_time:.0f} s")

    # ---- the whole idea in a few lines -------------------------------------------------
    obs = env.reset()                                  # every drone: start, goal, hidden problems
    controller.reset(env.n)
    rec = {k: [] for k in ("pos", "quat", "dist", "c_hat", "c_true", "tau_hat", "tau_true",
                           "alarm_lin", "alarm_ang", "eta")}
    crash_step = np.full(env.n, -1)
    for t in range(env.max_steps):
        action = controller.act(obs, env)              # (thrust, roll rate, pitch rate, yaw rate)
        obs, reward, crashed, timed_out, info = env.step(action)
        crash_step[crashed] = t
        rec["pos"].append(env.state["p"].copy())
        rec["quat"].append(env.state["q"].copy())
        rec["dist"].append(np.where(env.finished & (crash_step >= 0) & (crash_step < t), np.nan, env.dist))
        rec["c_hat"].append(env.cal_lin.theta[:, 0].copy())
        rec["c_true"].append(env.truth["c"].copy())
        rec["tau_hat"].append(env.cal_ang.theta.copy())
        rec["tau_true"].append(env.truth["tau"].copy())
        rec["alarm_lin"].append(env.cal_lin.alarm.copy())
        rec["alarm_ang"].append(env.cal_ang.alarm.copy())
        rec["eta"].append(env.eta.copy())
    # -------------------------------------------------------------------------------------
    rec = {k: np.array(v) for k, v in rec.items()}
    rec.update(goal=env.goal.copy(), crash_step=crash_step, event_step=scenario.event_step, dt=env.dt)

    e = scenario.event_step
    for i in range(env.n):
        d = rec["dist"][:, i]
        if crash_step[i] >= 0:
            print(f"  flight {i}: crashed at t = {crash_step[i] * env.dt:4.1f} s")
            continue
        alarms = int(rec["alarm_lin"][:, i].sum() + rec["alarm_ang"][:, i].sum())
        print(f"  flight {i}: before the change {d[e - 50:e].mean():.3f} m | worst after {np.nanmax(d[e:]):.3f} m | "
              f"last 2 s {d[-100:].mean():.3f} m | change detections {alarms}")

    os.makedirs(args.out, exist_ok=True)
    plot_trajectories_3d({label: rec}, 0, os.path.join(args.out, "trajectory.png"))
    print(f"saved {args.out}/trajectory.png")
    if not args.no_gif:
        make_gif(rec, 0, label, os.path.join(args.out, "episode.gif"),
                 title=f"scenario: {scenario.text}")
        print(f"saved {args.out}/episode.gif")


if __name__ == "__main__":
    main()
