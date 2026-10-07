"""Train a SAC agent to fly the drone.

    python train.py --run-name sac_calib                           # the self-calibrating agent
    python train.py --run-name sac_blind --set observation.mode=base
    python train.py --run-name smoke --set train.total_steps=30000 # one-minute check

Writes results/<run-name>/: config.yaml, train_log.csv (one row per finished episode),
eval_log.csv (one row per periodic evaluation), best.pt (the model) and last.pt.
"""
import argparse
import os
import time

import numpy as np
import torch

from drone_cal.env import DroneEnv
from drone_cal.evaluation import training_eval
from drone_cal.sac import ReplayBuffer, SACAgent, SACController
from drone_cal.scenarios import TrainingSampler
from drone_cal.utils import CSVLogger, get_device, load_config, save_config, set_seed


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config keys, e.g. observation.mode=base")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.total_steps is not None:
        cfg["train"]["total_steps"] = args.total_steps
    if args.device is not None:
        cfg["device"] = args.device
    tc = cfg["train"]
    seed = int(cfg["seed"])
    set_seed(seed)
    device = get_device(cfg["device"])
    torch.set_num_threads(2)          # several runs share the machine; more threads only contend

    out = os.path.join("results", args.run_name)
    os.makedirs(out, exist_ok=True)
    save_config(cfg, os.path.join(out, "config.yaml"))

    env = DroneEnv(cfg, tc["num_envs"], seed=seed)
    # held-out evaluation drones: same training distribution, different random stream
    eval_env = DroneEnv(cfg, tc["eval_episodes"], seed=10_000 + seed,
                        sampler=TrainingSampler(cfg), auto_reset=False)
    agent = SACAgent(env.obs_dim, env.action_dim, cfg["sac"], device)
    buf = ReplayBuffer(env.obs_dim, env.action_dim, cfg["sac"]["buffer_size"])
    rng = np.random.default_rng(seed + 1)
    train_log = CSVLogger(os.path.join(out, "train_log.csv"))
    eval_log = CSVLogger(os.path.join(out, "eval_log.csv"))

    print(f"run {args.run_name}: obs mode {env.obs_mode} ({env.obs_dim} numbers), "
          f"uncertainty {'on' if cfg['uncertainty']['enabled'] else 'off'}, "
          f"trigger {'on' if cfg['calibrator']['trigger'] else 'off'}, device {device}")

    obs = env.reset()
    n = env.n
    total = int(tc["total_steps"])
    step = 0
    next_eval = int(tc["eval_every"])
    best = -np.inf
    t0 = time.time()
    stats = {}
    recent = []
    while step < total:
        if step < tc["start_steps"]:
            act = rng.uniform(-1, 1, (n, env.action_dim)).astype(np.float32)
        else:
            act = agent.act(obs)
        next_obs, rew, term, trunc, info = env.step(act)
        store_next = next_obs.copy()
        if "done_idx" in info:                 # an episode ended: store the real last observation
            store_next[info["done_idx"]] = info["final_obs"]
        # time-limit truncation is not a real end: bootstrap through it (done = 0)
        buf.add_batch(obs, act, rew, store_next, term.astype(np.float32))
        obs = next_obs
        step += n

        for ep in info.get("episodes", []):
            recent.append(ep)
            train_log.log({"step": step, **ep, "alpha": stats.get("alpha", ""),
                           "loss_q": stats.get("loss_q", "")})
        if info.get("episodes"):
            stats = agent.stats()

        if step >= tc["learning_starts"]:
            for _ in range(int(tc["updates_per_step"])):
                agent.update(buf.sample(int(cfg["sac"]["batch_size"]), rng, device))

        if step >= next_eval or step >= total:
            next_eval += int(tc["eval_every"])
            stats = agent.stats()
            ev = training_eval(eval_env, SACController(agent))
            elapsed = time.time() - t0
            row = {"step": step, **ev, "alpha": stats.get("alpha", float("nan")),
                   "entropy": stats.get("entropy", float("nan")), "elapsed_s": round(elapsed, 1)}
            eval_log.log(row)
            tr = recent[-200:]
            tr_ret = np.mean([e["return"] for e in tr]) if tr else float("nan")
            tr_crash = np.mean([e["crashed"] for e in tr]) if tr else float("nan")
            tag = ""
            score = ev["return"]
            if score > best:
                best = score
                agent.save(os.path.join(out, "best.pt"), {"step": step, "eval": ev})
                tag = "  (best)"
            print(f"step {step:>8d} | train return {tr_ret:7.1f} crash {tr_crash:4.2f} | "
                  f"eval return {ev['return']:7.1f} crash {ev['crash_rate']:4.2f} "
                  f"success {ev['success_rate']:4.2f} final dist {ev['final_dist']:5.2f} m | "
                  f"alpha {row['alpha']:.3f} | {elapsed / 60:5.1f} min{tag}", flush=True)
    agent.save(os.path.join(out, "last.pt"), {"step": step})
    print(f"done in {(time.time() - t0) / 60:.1f} min; best eval return {best:.1f}")


if __name__ == "__main__":
    main()
