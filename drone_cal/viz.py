"""All figures and the animation.

Colours follow the controller, never its rank, so a controller has the same colour in
every figure. The seven hues were checked for colour-vision deficiency separation;
line style is a second cue (dashed = classical controller, dotted = trigger switched off).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .dynamics import quat_to_rot  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
EVENT = "#8a8984"

STYLE = {   # label -> (colour, line style)
    "SAC + calibrator": ("#2a78d6", "-"),
    "SAC, blind": ("#eb6834", "-"),
    "SAC + history": ("#1baf7a", "-"),
    "SAC + true parameters": ("#eda100", "-"),
    "PID": ("#e87ba4", "--"),
    "PID + calibrator": ("#008300", "--"),
    "SAC, nominal training": ("#4a3aa7", "-"),
    "SAC + calibrator, no trigger": ("#2a78d6", ":"),
    "PID + calibrator, no trigger": ("#008300", ":"),
}

SCENARIO_TITLES = {
    "nominal": "Nominal (no change)",
    "payload": "Payload +45 %",
    "motor_fault": "Motor -35 %",
    "wind_gust": "Wind 5 m/s",
    "combined": "All three at once",
    "severe": "Severe (beyond training)",
}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 9.5,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })


def _style(label):
    return STYLE.get(label, (INK_3, "-"))


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ training curves
def plot_training(run_dirs, labels, path):
    """Evaluation return and crash rate against training steps, one line per run."""
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    seen = set()
    for run, label in zip(run_dirs, labels):
        f = os.path.join(run, "eval_log.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        c, ls = _style(label)
        lab = None if label in seen else label
        seen.add(label)
        axes[0].plot(d["step"] / 1e6, d["return"], color=c, ls=ls, lw=1.6, alpha=0.9, label=lab)
        axes[1].plot(d["step"] / 1e6, 100 * d["crash_rate"], color=c, ls=ls, lw=1.6, alpha=0.9)
    axes[0].set_title("Evaluation return (higher is better)")
    axes[1].set_title("Crash rate on held-out flights, %")
    for ax in axes:
        ax.set_xlabel("training steps (millions)")
    axes[1].set_ylim(bottom=0)
    fig.legend(loc="lower center", ncol=min(len(seen), 5), bbox_to_anchor=(0.5, -0.08))
    _save(fig, path)


# ------------------------------------------------------------------ recovery curves
def plot_recovery(curves, labels, scenarios, controllers, dt, event_step, path):
    """Distance to goal over time per scenario: median line, interquartile band."""
    setup()
    cols = 3
    rows = int(np.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.3 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    inv = {v: k for k, v in labels.items()}
    for ax, sc in zip(axes, scenarios):
        n_ann = 0
        for label in controllers:
            key = f"{inv[label]}__{sc}__q"
            if key not in curves:
                continue
            q = curves[key]
            t = np.arange(q.shape[1]) * dt - event_step * dt
            c, ls = _style(label)
            ax.fill_between(t, q[0], q[2], color=c, alpha=0.10, lw=0)
            crash = float(curves[f"{inv[label]}__{sc}__crashed"][0])
            ax.plot(t, q[1], color=c, ls=ls, label=label if sc == scenarios[0] else None)
            if crash > 0:
                ax.text(0.98, 0.97 - 0.07 * n_ann, f"{label}: {100 * crash:.0f}% crashed", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5, color=INK_2)
                n_ann += 1
        ax.axvline(0, color=EVENT, lw=1, ls=(0, (4, 3)))
        ax.set_title(SCENARIO_TITLES.get(sc, sc))
        ax.set_yscale("log")
        ax.set_ylim(0.005, 5)
        ax.set_xlim(-2, t[-1])
    for ax in axes[len(scenarios):]:
        ax.axis("off")
    for ax in axes[::cols]:
        ax.set_ylabel("distance to goal, m (log)")
    for ax in axes[-cols:]:
        ax.set_xlabel("seconds since the change")
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, 1.0, "Lines: median over episodes that did not crash. Bands: middle 50 %. "
             "Dashed grey line: the moment uncertainty arrives.", ha="center", color=INK_2, fontsize=9)
    _save(fig, path)


# ------------------------------------------------------------------ scenario bars
def plot_scenario_bars(summary, controllers, scenarios, path):
    """Two panels (never two scales on one axis): error after the change, and crash rate."""
    setup()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
    k = len(controllers)
    width = 0.8 / k
    x = np.arange(len(scenarios))
    for j, label in enumerate(controllers):
        c, ls = _style(label)
        sub = summary[summary.controller == label].set_index("scenario").reindex(scenarios)
        xs = x - 0.4 + width * (j + 0.5)
        hatch = "///" if ls == "--" else ("..." if ls == ":" else None)
        rm = sub["rmse_post"].to_numpy()
        axes[0].bar(xs, np.nan_to_num(rm, nan=0.0), width * 0.86, color=c, label=label,
                    hatch=hatch, edgecolor=SURFACE, linewidth=0)
        for xi, v in zip(xs, rm):
            if np.isnan(v):
                axes[0].text(xi, 0.02, "all\ncrashed", ha="center", va="bottom", fontsize=7, color=INK_2,
                             transform=axes[0].get_xaxis_transform())
        axes[1].bar(xs, 100 * sub["crash_rate"].to_numpy(), width * 0.86, color=c, hatch=hatch,
                    edgecolor=SURFACE, linewidth=0)
        if "crash_rate_sd" in sub and sub["seeds"].max() > 1:
            axes[1].errorbar(xs, 100 * sub["crash_rate"], yerr=100 * sub["crash_rate_sd"].fillna(0),
                             fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
            axes[0].errorbar(xs, sub["rmse_post"], yerr=sub["rmse_post_sd"].fillna(0),
                             fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMS distance after the change, m (log)")
    axes[0].set_title("How far each controller is pushed off its goal (lower is better; crashed flights excluded)")
    axes[1].set_ylabel("crashed, %")
    axes[1].set_title("How often it crashes")
    axes[1].set_xticks(x, [SCENARIO_TITLES.get(s, s) for s in scenarios])
    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    _save(fig, path)


# ------------------------------------------------------------------ calibration traces
def plot_calibration(rec_by_variant, ep, dt, event_step, path, c0=1.0):
    """Estimates against the truth for one flight, one column per estimator variant."""
    setup()
    variants = list(rec_by_variant)
    fig, axes = plt.subplots(3, len(variants), figsize=(4.2 * len(variants), 7.2), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    if axes.shape[0] != 3:
        axes = axes.T
    for j, v in enumerate(variants):
        r = rec_by_variant[v]
        T = r["c_hat"].shape[0]
        t = np.arange(T) * dt - event_step * dt
        rows = [
            ("thrust gain c, 1/kg", r["c_true"][:, ep], r["c_hat"][:, ep], r["c_std"][:, ep]),
            ("disturbance d_x, m/s²", r["d_true"][:, ep, 0], r["d_hat"][:, ep, 0], r["d_std"][:, ep, 0]),
            ("fault torque τ_x, N m", r["tau_true"][:, ep, 0], r["tau_hat"][:, ep, 0], r["tau_std"][:, ep, 0]),
        ]
        alarms = [np.flatnonzero(r["alarm_lin"][:, ep]), np.flatnonzero(r["alarm_lin"][:, ep]),
                  np.flatnonzero(r["alarm_ang"][:, ep])]
        for i, (name, truth, est, std) in enumerate(rows):
            ax = axes[i, j]
            ax.plot(t, truth, color=INK, lw=1.4, label="true value" if (i, j) == (0, 0) else None)
            ax.fill_between(t, est - 2 * std, est + 2 * std, color="#2a78d6", alpha=0.15, lw=0,
                            label="± 2 σ (its own uncertainty)" if (i, j) == (0, 0) else None)
            ax.plot(t, est, color="#2a78d6", lw=1.6, label="estimate" if (i, j) == (0, 0) else None)
            for a in alarms[i]:
                ax.axvline(t[a], color="#e34948", lw=1.0, alpha=0.8)
            ax.axvline(0, color=EVENT, lw=1, ls=(0, (4, 3)))
            if j == 0:
                ax.set_ylabel(name)
            if i == 0:
                ax.set_title(v)
        axes[-1, j].set_xlabel("seconds since the change")
        axes[0, j].set_xlim(-1.5, 3.0)
        axes[1, j].set_ylim(-1.5, 1.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=INK, lw=1.4), Line2D([], [], color="#2a78d6", lw=1.6),
               Patch(color="#2a78d6", alpha=0.15), Line2D([], [], color="#e34948", lw=1.0),
               Line2D([], [], color=EVENT, lw=1, ls=(0, (4, 3)))]
    names = ["true value", "estimate", "± 2 σ (its own uncertainty)", "change detected",
             "uncertainty arrives"]
    fig.legend(handles, names, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    _save(fig, path)


# ------------------------------------------------------------------ 3D trajectories
def _set_box(ax, center, half):
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(max(center[2] - half, 0), center[2] + half)


def plot_trajectories_3d(traces, ep, path):
    """Same flight, same fault, different controllers: where each drone went after the change.
    Each panel is zoomed to its own flight, so read the scale on the axes and the title."""
    setup()
    fig = plt.figure(figsize=(6.5 * len(traces), 5.4))
    for j, (label, r) in enumerate(traces.items()):
        ax = fig.add_subplot(1, len(traces), j + 1, projection="3d")
        P = r["pos"][:, ep]
        dist = r["dist"][:, ep]
        T = np.flatnonzero(~np.isnan(dist))[-1] + 1
        e = int(r["event_step"])
        c, _ = _style(label)
        g = r["goal"][ep]
        s0 = max(e - int(round(1.0 / float(r["dt"]))), 0)        # one second before the change
        ax.plot(*P[s0:min(e, T)].T, color=INK_3, lw=1.4, label="the second before the change")
        ax.plot(*P[e:T].T, color=c, lw=1.8, label="after the change")
        ax.scatter(*g, color=INK, marker="*", s=120, label="goal")
        ax.scatter(*P[min(e, T - 1)], color="#e34948", s=30, label="uncertainty arrives")
        crashed = r["crash_step"][ep] >= 0
        if crashed:
            ax.scatter(*P[T - 1], color=INK, marker="x", s=60, label="crash")
        half = max(0.3, np.nanmax(np.abs(P[s0:T] - g)) + 0.1)
        ax.set_xlim(g[0] - half, g[0] + half)
        ax.set_ylim(g[1] - half, g[1] + half)
        ax.set_zlim(g[2] - half, g[2] + half)
        worst = np.nanmax(dist[e:T])
        ax.set_title(f"{label}\nworst distance after the change {worst:.2f} m" + ("  (crashed)" if crashed else ""))
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        ax.set_zlabel("z, m")
        ax.view_init(elev=22, azim=-60)
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


# ------------------------------------------------------------------ animation
def _drone_lines(p, q, scale):
    R = quat_to_rot(q[None])[0]
    d = 0.17 / np.sqrt(2) * scale
    motors = np.array([[d, d, 0], [-d, d, 0], [-d, -d, 0], [d, -d, 0]]) @ R.T + p
    return motors, R


def make_gif(rec, ep, label, path, scale=3.0, stride=3, fps=20, title=None):
    """3D view of one flight next to the calibrator's estimates, frame by frame."""
    import imageio.v2 as imageio
    setup()
    P = rec["pos"][:, ep]
    Q = rec["quat"][:, ep]
    dist = rec["dist"][:, ep]
    T = np.flatnonzero(~np.isnan(dist))[-1] + 1
    e = int(rec["event_step"])
    dt = float(rec["dt"])
    g = rec["goal"][ep]
    t = np.arange(T) * dt
    c_hat, c_true = rec["c_hat"][:T, ep], rec["c_true"][:T, ep]
    tau_hat = np.linalg.norm(rec["tau_hat"][:T, ep, :2], axis=1)
    tau_true = np.linalg.norm(rec["tau_true"][:T, ep, :2], axis=1)
    al = np.flatnonzero(rec["alarm_lin"][:T, ep] | rec["alarm_ang"][:T, ep])
    color, _ = _style(label)
    half = max(0.8, np.nanmax(np.abs(P[:T] - g)) + 0.3)
    frames = []
    fig = plt.figure(figsize=(10, 4.6))
    for k in list(range(0, T, stride)) + [T - 1]:
        fig.clf()
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot(*P[:k + 1].T, color=color, lw=1.2, alpha=0.7)
        ax.scatter(*g, color=INK, marker="*", s=90)
        motors, R = _drone_lines(P[k], Q[k], scale)
        for a, b in ((0, 2), (1, 3)):
            ax.plot(*np.stack([motors[a], motors[b]]).T, color=INK, lw=2.2)
        healthy = rec["eta"][k, ep]
        for i in range(4):
            mc = "#e34948" if healthy[i] < 0.99 else INK_2
            ax.scatter(*motors[i], color=mc, s=28)
        ax.plot(*np.stack([P[k], P[k] + R[:, 2] * 0.35 * scale / 3]).T, color=color, lw=1.2)
        _set_box(ax, g, half)
        ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
        ax.view_init(elev=20, azim=-60 + 0.15 * k)
        status = "UNCERTAINTY ARRIVED" if k >= e else "nominal"
        ax.set_title(f"{label}   t = {t[k]:4.1f} s   distance {dist[k]:.2f} m\n{status}",
                     fontsize=10, color="#e34948" if k >= e else INK)

        ax1 = fig.add_subplot(2, 2, 2)
        ax1.plot(t[:k + 1], c_true[:k + 1], color=INK, lw=1.3, label="true")
        ax1.plot(t[:k + 1], c_hat[:k + 1], color="#2a78d6", lw=1.6, label="estimated")
        ax1.set_xlim(0, t[-1]), ax1.set_ylim(min(c_true.min(), c_hat.min()) - 0.05, max(c_true.max(), c_hat.max()) + 0.05)
        ax1.set_title("thrust gain, 1/kg (falls with payload, damage)", fontsize=9)
        ax1.legend(loc="lower left", fontsize=8)
        ax1.tick_params(labelbottom=False)
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.plot(t[:k + 1], tau_true[:k + 1], color=INK, lw=1.3)
        ax2.plot(t[:k + 1], tau_hat[:k + 1], color="#2a78d6", lw=1.6)
        ax2.set_xlim(0, t[-1]), ax2.set_ylim(-0.01, max(tau_true.max(), tau_hat.max()) + 0.03)
        ax2.set_title("fault torque, N m (a weak motor tilts the drone)", fontsize=9)
        ax2.set_xlabel("time, s")
        for axx in (ax1, ax2):
            axx.axvline(e * dt, color=EVENT, lw=1, ls=(0, (4, 3)))
            for a in al[al <= k]:
                axx.axvline(t[a], color="#e34948", lw=0.8, alpha=0.7)
        if title:
            fig.suptitle(title, fontsize=10, color=INK_2)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
