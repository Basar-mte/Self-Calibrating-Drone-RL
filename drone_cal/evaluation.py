"""Rollouts and metrics.

Every controller (a trained agent or the PID) is flown on the same drones: the same
seed gives the same starts, goals, faults and wind, so differences come from the controller.

Metrics, computed per episode and then summarised (distances are to the goal, in metres):
    crash        the drone hit the ground, flipped, or drifted more than 6 m away
    pre          mean distance in the second before the event (how well it was holding)
    peak         largest distance in the 3 s after the event (how far the change knocks it)
    rmse_post    root-mean-square distance from the event to the end
    settle       time after the event until the distance stays below 0.25 m for good
                 (infinite if it never does)
    hold         mean distance over the last 2 s
    success      no crash and hold < 0.3 m
The averages of pre / peak / rmse_post / hold are over the episodes that did not crash;
the crash rate is always reported next to them.
"""
import numpy as np

SETTLE_TOL = 0.25
SUCCESS_HOLD = 0.3


def run_episodes(env, controller, record=True):
    """Fly one full episode on every drone of a non-auto-resetting env."""
    assert not env.auto_reset, "evaluation needs auto_reset=False"
    obs = env.reset()
    controller.reset(env.n)
    T, n = env.max_steps, env.n
    rec = {
        "dist": np.full((T, n), np.nan),
        "crash_step": np.full(n, -1),
        "returns": np.zeros(n),
    }
    if record:
        rec.update({
            "pos": np.zeros((T, n, 3)), "goal": env.goal.copy(),
            "quat": np.zeros((T, n, 4)), "action": np.zeros((T, n, 4)),
            "c_hat": np.zeros((T, n)), "d_hat": np.zeros((T, n, 3)), "tau_hat": np.zeros((T, n, 3)),
            "c_std": np.zeros((T, n)), "d_std": np.zeros((T, n, 3)), "tau_std": np.zeros((T, n, 3)),
            "c_true": np.zeros((T, n)), "d_true": np.zeros((T, n, 3)), "tau_true": np.zeros((T, n, 3)),
            "alarm_lin": np.zeros((T, n), dtype=bool), "alarm_ang": np.zeros((T, n), dtype=bool),
            "eta": np.zeros((T, n, 4)), "mass": np.zeros((T, n)), "wind": np.zeros((T, n, 3)),
        })
    for t in range(T):
        was_finished = env.finished.copy()
        a = controller.act(obs, env)
        obs, r, term, trunc, info = env.step(a)
        rec["returns"] += r
        live = ~was_finished
        rec["dist"][t, live] = env.dist[live]
        rec["crash_step"][term] = t
        if record:
            rec["pos"][t] = env.state["p"]
            rec["quat"][t] = env.state["q"]
            rec["action"][t] = np.clip(a, -1, 1)
            rec["c_hat"][t] = env.cal_lin.theta[:, 0]
            rec["d_hat"][t] = env.cal_lin.theta[:, 1:4]
            rec["tau_hat"][t] = env.cal_ang.theta
            std_l = env.cal_lin.std
            rec["c_std"][t] = std_l[:, 0]
            rec["d_std"][t] = std_l[:, 1:4]
            rec["tau_std"][t] = env.cal_ang.std
            rec["c_true"][t] = env.truth["c"]
            rec["d_true"][t] = env.truth["d"]
            rec["tau_true"][t] = env.truth["tau"]
            rec["alarm_lin"][t] = env.cal_lin.alarm
            rec["alarm_ang"][t] = env.cal_ang.alarm
            rec["eta"][t] = env.eta
            rec["mass"][t] = env.mass
            rec["wind"][t] = env.wind_mean + env.gust
        if env.finished.all():
            break
    return rec


def episode_metrics(dist, crash_step, event_step, dt):
    """Per-episode metrics from a (T, n) distance array. Rows after a crash are NaN."""
    T, n = dist.shape
    e = int(event_step)
    w1, w2, w3 = int(round(1.0 / dt)), int(round(2.0 / dt)), int(round(3.0 / dt))
    out = {k: np.full(n, np.nan) for k in ("pre", "peak", "rmse_post", "settle", "hold")}
    crashed = crash_step >= 0
    for i in np.flatnonzero(~crashed):
        d = dist[:, i]
        post = d[e:]
        out["pre"][i] = d[max(e - w1, 0):e].mean()
        out["peak"][i] = d[e:e + w3].max()
        out["rmse_post"][i] = np.sqrt(np.mean(post ** 2))
        out["hold"][i] = d[-w2:].mean()
        above = np.flatnonzero(post > SETTLE_TOL)
        if above.size == 0:
            out["settle"][i] = 0.0
        elif above[-1] == post.size - 1:
            out["settle"][i] = np.inf
        else:
            out["settle"][i] = (above[-1] + 1) * dt
    out["crashed"] = crashed
    out["success"] = (~crashed) & (out["hold"] < SUCCESS_HOLD)
    return out


def summarise(m):
    ok = ~m["crashed"]
    settle = m["settle"][ok]
    finite = settle[np.isfinite(settle)]
    nan_mean = lambda x: float(np.mean(x)) if x.size else float("nan")
    return {
        "episodes": int(m["crashed"].size),
        "crash_rate": float(m["crashed"].mean()),
        "success_rate": float(m["success"].mean()),
        "pre": nan_mean(m["pre"][ok]),
        "peak": nan_mean(m["peak"][ok]),
        "rmse_post": nan_mean(m["rmse_post"][ok]),
        "hold": nan_mean(m["hold"][ok]),
        "settled_rate": float(np.isfinite(m["settle"]).mean()),
        "settle_median": float(np.median(settle)) if settle.size else float("nan"),
        "settle_mean_settled": nan_mean(finite),
    }


def training_eval(env, controller):
    """Short evaluation used while training: mean return, crash rate, final distance."""
    rec = run_episodes(env, controller, record=False)
    last = np.array([rec["dist"][~np.isnan(rec["dist"][:, i]), i][-1] for i in range(env.n)])
    crashed = rec["crash_step"] >= 0
    return {
        "return": float(rec["returns"].mean()),
        "crash_rate": float(crashed.mean()),
        "final_dist": float(np.mean(last[~crashed])) if (~crashed).any() else float("nan"),
        "success_rate": float(np.mean((~crashed) & (last < SUCCESS_HOLD))),
    }
