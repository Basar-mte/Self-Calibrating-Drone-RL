"""How fast and how cleanly does the calibrator re-identify the drone after a change?

    python calibration_study.py                 # about a minute

The estimator is studied on its own, separated from any controller that uses it:

  1. Fly 100 drones with the plain PID (which ignores the calibrator), once in the
     "combined" scenario and once in the "nominal" one, and record exactly what the
     calibrator was given at every step (regressors and measurements) plus the truth.
  2. Replay those identical recordings through three estimator designs:
        slow       Kalman filter that trusts its estimate (small drift), no change detector
        fast       the same filter forgetting ten times faster, no change detector
        triggered  the slow filter plus the CUSUM change detector (the default)
  3. Measure, for each design:
        convergence time   after the change, how long until the thrust-gain estimate is
                           within 5 % of the truth and the fault-torque estimate within
                           0.02 N m, and both stay there
        error spread       standard deviation of the thrust-gain error (in %) during steady
                           nominal hover, over time and flights
        false alarms       change detections per minute in the nominal flights

Writes results/calibration/summary.json and calibration.png.
"""
import argparse
import json
import os

import numpy as np

from drone_cal.calibrator import make_calibrators
from drone_cal.env import DroneEnv
from drone_cal.pid import CascadedPID
from drone_cal.scenarios import EvalScenario
from drone_cal.utils import deep_copy, load_config
from drone_cal.viz import plot_calibration

VARIANTS = {
    "slow, no detector": dict(drift_scale=1.0, trigger=False),
    "fast forgetting, no detector": dict(drift_scale=10.0, trigger=False),
    "slow + change detector (default)": dict(drift_scale=1.0, trigger=True),
}


def record_flight(cfg, scenario, n, seed):
    env = DroneEnv(cfg, n, seed=seed, sampler=EvalScenario(scenario, cfg), auto_reset=False, obs_mode="base")
    ctl = CascadedPID(cfg, n, use_calibration=False)
    obs = env.reset()
    ctl.reset(n)
    T = env.max_steps
    data = {"Phi_lin": np.zeros((T, n, 3, 4)), "y_lin": np.zeros((T, n, 3)), "y_ang": np.zeros((T, n, 3)),
            "c_true": np.zeros((T, n)), "d_true": np.zeros((T, n, 3)), "tau_true": np.zeros((T, n, 3))}
    crashed = np.zeros(n, dtype=bool)
    for t in range(T):
        obs, _, term, _, _ = env.step(ctl.act(obs, env))
        crashed |= term
        data["Phi_lin"][t] = env.cal_inputs["Phi_lin"]
        data["y_lin"][t] = env.cal_inputs["y_lin"]
        data["y_ang"][t] = env.cal_inputs["y_ang"]
        data["c_true"][t] = env.truth["c"]
        data["d_true"][t] = env.truth["d"]
        data["tau_true"][t] = env.truth["tau"]
    data["complete"] = ~crashed                       # flights that never crashed
    return data, env.dt, EvalScenario(scenario, cfg).event_step, env.quad.mass


def replay(cfg_cal, data, mass, drift_scale, trigger):
    c = deep_copy(cfg_cal)
    c["thrust_gain_drift"] *= drift_scale
    c["disturbance_drift"] = [x * drift_scale for x in c["disturbance_drift"]]
    c["torque_drift"] *= drift_scale
    T, n = data["y_lin"].shape[:2]
    lin, ang = make_calibrators(n, c, mass, trigger=trigger)
    out = {k: np.zeros((T, n)) for k in ("c_hat", "c_std")}
    out.update({k: np.zeros((T, n, 3)) for k in ("d_hat", "d_std", "tau_hat", "tau_std")})
    out.update({k: np.zeros((T, n), dtype=bool) for k in ("alarm_lin", "alarm_ang")})
    eye = np.tile(np.eye(3), (n, 1, 1))
    for t in range(T):
        lin.update(data["Phi_lin"][t], data["y_lin"][t])
        ang.update(eye, data["y_ang"][t])
        out["c_hat"][t] = lin.theta[:, 0]
        out["d_hat"][t] = lin.theta[:, 1:4]
        out["tau_hat"][t] = ang.theta
        s = lin.std
        out["c_std"][t] = s[:, 0]
        out["d_std"][t] = s[:, 1:4]
        out["tau_std"][t] = ang.std
        out["alarm_lin"][t] = lin.alarm
        out["alarm_ang"][t] = ang.alarm
    for k in ("c_true", "d_true", "tau_true"):
        out[k] = data[k]
    return out


def convergence_time(err, tol, start, dt):
    """Per episode: seconds after `start` until err stays below tol (inf if never)."""
    post = err[start:]
    res = np.full(err.shape[1], np.inf)
    for i in range(err.shape[1]):
        above = np.flatnonzero(post[:, i] > tol)
        if above.size == 0:
            res[i] = 0.0
        elif above[-1] < post.shape[0] - 1:
            res[i] = (above[-1] + 1) * dt
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="results/calibration")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    flights = {name: record_flight(cfg, name, args.episodes, seed=70_000 + k)
               for k, name in enumerate(["combined", "nominal"])}
    summary = {}
    traces = {}
    for vname, v in VARIANTS.items():
        row = {}
        for name, (data, dt, ev, mass) in flights.items():
            r = replay(cfg["calibrator"], data, mass, v["drift_scale"], v["trigger"])
            ok = data["complete"]
            minutes = data["y_lin"].shape[0] * dt / 60.0
            if name == "combined":
                traces[vname] = r
                c_err = np.abs(r["c_hat"] - r["c_true"]) / r["c_true"]
                tau_err = np.linalg.norm(r["tau_hat"] - r["tau_true"], axis=2)
                tc = convergence_time(c_err[:, ok], 0.05, ev, dt)
                tt = convergence_time(tau_err[:, ok], 0.02, ev, dt)
                both = np.maximum(tc, tt)
                row.update({
                    "flights": int(ok.sum()),
                    "thrust_gain_converge_s_median": float(np.median(tc)),
                    "fault_torque_converge_s_median": float(np.median(tt)),
                    "both_converge_s_median": float(np.median(both)),
                    "both_converge_s_p90": float(np.percentile(both, 90)),
                    "never_converged": float(np.mean(~np.isfinite(both))),
                })
            else:
                pre = slice(int(1.0 / dt), ev)        # steady hover before the (empty) event
                c_err = (r["c_hat"] - r["c_true"]) / r["c_true"]
                row["thrust_gain_error_spread_pct"] = float(100 * np.std(c_err[pre][:, ok]))
                row["false_alarms_per_min"] = float((r["alarm_lin"] | r["alarm_ang"])[:, ok].sum(0).mean() / minutes)
        summary[vname] = row
        print(f"{vname:34s} " + "  ".join(f"{k} {val:.3g}" for k, val in row.items()))

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    data, dt, ev, _ = flights["combined"]
    ep = int(np.flatnonzero(data["complete"])[0])
    plot_calibration(traces, ep, dt, ev, os.path.join(args.out, "calibration.png"))
    print(f"wrote {args.out}/summary.json and calibration.png")


if __name__ == "__main__":
    main()
