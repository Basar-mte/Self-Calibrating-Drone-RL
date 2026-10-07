"""Sanity checks. Run with:  python tests/test_env.py   (about 10 seconds; prints one "ok" per check)

Each check tests one physical or mathematical fact that everything else relies on.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drone_cal.calibrator import KalmanCalibrator  # noqa: E402
from drone_cal.dynamics import GRAVITY, Quadrotor, quat_to_rot  # noqa: E402
from drone_cal.env import DroneEnv  # noqa: E402
from drone_cal.evaluation import episode_metrics, run_episodes  # noqa: E402
from drone_cal.pid import CascadedPID  # noqa: E402
from drone_cal.scenarios import EvalScenario  # noqa: E402
from drone_cal.utils import load_config  # noqa: E402

CFG = load_config(os.path.join(ROOT, "configs", "default.yaml"))


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAILED: {name} {detail}")
    print(f"ok  {name}")


def test_mixer_inverts():
    q = Quadrotor(CFG["drone"])
    f = np.array([[2.0, 2.5, 3.0, 1.5]])
    u = f @ q.alloc.T
    back = u @ q.alloc_inv.T
    check("mixer: motor thrusts -> (thrust, torques) -> motor thrusts round trip", np.allclose(f, back))


def test_hover_is_still():
    """A level drone with exactly hover thrust and zero rate command must stay put."""
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"),
                      [f"sensors.{k}=0.0" for k in CFG["sensors"]])
    env = DroneEnv(cfg, 4, seed=0, sampler=EvalScenario("nominal", cfg), auto_reset=False)
    env.reset()
    env.state["q"][:] = [1, 0, 0, 0]
    env.state["v"][:] = 0
    env.state["w"][:] = 0
    p0 = env.state["p"].copy()
    for _ in range(100):
        env.step(np.zeros((4, 4)))
    drift = np.abs(env.state["p"] - p0).max()
    check("physics: hover thrust holds a level drone in place for 2 s", drift < 1e-6, f"(drift {drift:.2e} m)")


def test_quaternion_stays_unit():
    env = DroneEnv(CFG, 8, seed=1)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(200):
        env.step(rng.uniform(-1, 1, (8, 4)))
    norms = np.linalg.norm(env.state["q"], axis=1)
    R = quat_to_rot(env.state["q"])
    orth = np.abs(R @ np.transpose(R, (0, 2, 1)) - np.eye(3)).max()
    check("rotations: quaternions stay unit length, rotation matrices orthonormal",
          np.allclose(norms, 1) and orth < 1e-9)


def test_free_fall():
    """With the motors off a drone falls at g (drag aside)."""
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"), ["drone.linear_drag=0.0"])
    env = DroneEnv(cfg, 1, seed=0, sampler=EvalScenario("nominal", cfg), auto_reset=False)
    env.reset()
    env.state["f"][:] = 0
    env.state["v"][:] = 0
    env.state["q"][:] = [1, 0, 0, 0]
    env.state["w"][:] = 0
    env.step(np.array([[-1.0, 0, 0, 0]]))   # zero thrust command; motors were already off
    a = env.state["v"][0, 2] / env.dt
    check("physics: free fall accelerates at g", abs(a + GRAVITY) < 0.05, f"(got {a:.3f})")


def test_calibrator_recovers_parameters():
    """Feed synthetic data y = Phi theta + noise; the estimate must converge to theta."""
    rng = np.random.default_rng(0)
    cal = KalmanCalibrator(1, theta0=[1.0, 0, 0, 0], prior_std=[0.3, 3, 3, 0.5],
                           drift_std=1e-5, meas_std=0.3, trigger=False)   # constant truth: tiny drift
    true = np.array([0.7, 1.0, -0.5, 0.0])
    for _ in range(400):
        Phi = np.zeros((1, 3, 4))
        Phi[0, :, 0] = np.array([0.05, -0.03, 1.0]) * rng.uniform(8, 12)
        Phi[0, :, 1:] = np.eye(3)
        y = Phi[0] @ true + rng.normal(0, 0.3, 3)
        cal.update(Phi, y[None])
    err = np.abs(cal.theta[0] - true)
    check("calibrator: converges to known parameters on synthetic data", err.max() < 0.1, f"(errors {err.round(3)})")


def test_detector_fires_on_change_only():
    """No false alarms on a steady system; an alarm within a few steps of a jump."""
    rng = np.random.default_rng(1)
    cal = KalmanCalibrator(1, theta0=[0, 0, 0], prior_std=0.15, drift_std=5e-4, meas_std=0.006)
    eye = np.eye(3)[None]
    alarms = []
    for t in range(1000):
        tau = np.zeros(3) if t < 500 else np.array([0.1, -0.1, 0.01])
        cal.update(eye, (tau + rng.normal(0, 0.006, 3))[None])
        alarms.append(cal.alarm[0])
    alarms = np.array(alarms)
    first = np.flatnonzero(alarms[500:])
    check("change detector: silent for 500 steady steps, fires within 3 steps of a jump",
          not alarms[:500].any() and first.size > 0 and first[0] <= 3)


def test_pid_calibrated_holds_through_fault():
    sc = EvalScenario("combined", CFG)
    env = DroneEnv(CFG, 10, seed=3, sampler=sc, auto_reset=False)
    rec = run_episodes(env, CascadedPID(CFG, 10, use_calibration=True), record=False)
    m = episode_metrics(rec["dist"], rec["crash_step"], sc.event_step, env.dt)
    check("closed loop: PID + calibrator survives the combined fault and holds within 0.1 m",
          (~m["crashed"]).all() and np.nanmax(m["hold"]) < 0.1, f"(hold {np.nanmax(m['hold']):.3f})")


def test_observation_sizes_and_seeding():
    sizes = {}
    for mode in ("base", "calib", "oracle", "history"):
        env = DroneEnv(CFG, 3, seed=7, obs_mode=mode)
        sizes[mode] = env.reset().shape[1] == env.obs_dim
    a = DroneEnv(CFG, 3, seed=7).reset()
    b = DroneEnv(CFG, 3, seed=7).reset()
    check("env: observation sizes match obs_dim in every mode; same seed, same drones",
          all(sizes.values()) and np.array_equal(a, b))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all checks passed")
