"""The simulated world: many drones flown in parallel, each with its own hidden problems.

One call to `step(actions)` advances every drone by one control period (dt = 0.02 s):

  1. if a drone's event time has come, its true mass / motor efficiencies / wind jump
  2. the wind gusts evolve (Ornstein-Uhlenbeck noise around the mean wind)
  3. the action (collective thrust, body rates) is held for `substeps` physics steps, each
     running the rate loop, the mixer, the motor lag and the rigid-body dynamics
  4. the sensors are read, with noise
  5. the calibrator updates its estimates from the accelerometer and gyro
  6. reward and termination are computed from the true state

The class follows the Gymnasium conventions (terminated / truncated), but is batched:
every array has the number of drones as its first dimension.

Observation modes (config `observation.mode`) decide what the agent sees:
    base     position error and velocity (coarse and fine scale), attitude, body rates,
             previous action                                                       (28)
    calib    base + calibrator estimates and their uncertainty                     (42)
    oracle   base + the true values the calibrator is trying to estimate           (35)
    history  base + the last H frames of velocity, rate, accelerometer and action  (28 + 13 H)

Position error and velocity appear twice: once scaled for flying across the arena, once
through tanh at a 0.3 m and 0.5 m/s scale, so that a 5 cm error is a clear signal
(0.17) instead of a tiny one (0.025) the network would struggle to act on precisely.
"""
import numpy as np

from .calibrator import make_calibrators
from .dynamics import GRAVITY, Quadrotor, quat_mul, quat_to_rot, rotvec_to_quat, yaw_of
from .scenarios import TrainingSampler

BASE_DIM = 28
CALIB_DIM = 14
ORACLE_DIM = 7
FRAME_DIM = 13
GUST_TIME_CONSTANT = 1.0      # s
GUST_AXIS_SCALE = np.array([1.0, 1.0, 0.3])


class DroneEnv:
    def __init__(self, cfg, num_envs=1, seed=0, sampler=None, auto_reset=True,
                 obs_mode=None, trigger=None):
        self.cfg = cfg
        self.n = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.quad = Quadrotor(cfg["drone"])
        self.dt = float(cfg["sim"]["dt"])
        self.substeps = int(cfg["sim"]["substeps"])
        self.sub_dt = self.dt / self.substeps
        self.sampler = sampler if sampler is not None else TrainingSampler(cfg)
        self.max_steps = self.sampler.max_steps
        self.auto_reset = auto_reset
        self.obs_mode = obs_mode or cfg["observation"]["mode"]
        if self.obs_mode not in ("base", "calib", "oracle", "history"):
            raise ValueError(f"unknown observation mode {self.obs_mode!r}")
        self.H = int(cfg["observation"]["history"])
        self.sens = cfg["sensors"]
        self.rw = cfg["reward"]
        self.lost_distance = float(cfg["task"]["lost_distance"])
        self.c0 = 1.0 / self.quad.mass
        self.cal_lin, self.cal_ang = make_calibrators(self.n, cfg["calibrator"], self.quad.mass, trigger)

        n = self.n
        self.state = {k: np.zeros((n, d)) for k, d in (("p", 3), ("v", 3), ("q", 4), ("w", 3), ("f", 4))}
        self.state["q"][:, 0] = 1.0
        self.mass = np.full(n, self.quad.mass)
        self.eta = np.ones((n, 4))
        self.wind_mean = np.zeros((n, 3))
        self.gust = np.zeros((n, 3))
        self.gust_std = np.zeros(n)
        self.goal = np.zeros((n, 3))
        self.steps = np.zeros(n, dtype=int)
        self.event_step = np.full(n, -1)
        self.post = {"mass": self.mass.copy(), "eta": self.eta.copy(),
                     "wind": self.wind_mean.copy(), "gust": self.gust_std.copy()}
        self.prev_action = np.zeros((n, 4))
        self.finished = np.zeros(n, dtype=bool)
        self.meas = {"p": np.zeros((n, 3)), "v": np.zeros((n, 3)), "R": np.tile(np.eye(3), (n, 1, 1)),
                     "w": np.zeros((n, 3)), "sf": np.zeros((n, 3))}
        self.truth = {"c": np.full(n, self.c0), "d": np.zeros((n, 3)), "tau": np.zeros((n, 3))}
        self.hist = np.zeros((n, self.H, FRAME_DIM))
        self.w_meas_prev = np.zeros((n, 3))
        self.dist = np.zeros(n)
        self.ep_return = np.zeros(n)
        self.cal_inputs = {}

    # ------------------------------------------------------------ sizes
    @property
    def obs_dim(self):
        extra = {"base": 0, "calib": CALIB_DIM, "oracle": ORACLE_DIM, "history": FRAME_DIM * self.H}
        return BASE_DIM + extra[self.obs_mode]

    action_dim = 4

    @property
    def time(self):
        return self.steps * self.dt

    # ------------------------------------------------------------ reset
    def reset(self):
        idx = np.arange(self.n)
        self._reset_idx(idx)
        return self._obs()

    def _reset_idx(self, idx):
        k = idx.size
        smp = self.sampler.sample(self.rng, k)
        s = self.state
        s["p"][idx] = smp["start_pos"]
        s["v"][idx] = smp["start_vel"]
        s["q"][idx] = smp["start_quat"]
        s["w"][idx] = smp["start_rate"]
        self.goal[idx] = smp["goal"]
        self.mass[idx] = smp["mass0"]
        self.eta[idx] = smp["eta0"]
        self.wind_mean[idx] = smp["wind0"]
        self.gust_std[idx] = smp["gust0"]
        self.gust[idx] = 0.0
        self.event_step[idx] = smp["event_step"]
        self.post["mass"][idx] = smp["mass1"]
        self.post["eta"][idx] = smp["eta1"]
        self.post["wind"][idx] = smp["wind1"]
        self.post["gust"][idx] = smp["gust1"]
        # motors start at the hover thrust of the real drone (it was hovering before the episode)
        s["f"][idx] = np.clip((smp["mass0"] * GRAVITY / 4.0)[:, None] / smp["eta0"], 0.0, self.quad.f_max)
        self.steps[idx] = 0
        self.prev_action[idx] = 0.0
        self.finished[idx] = False
        self.ep_return[idx] = 0.0
        self.cal_lin.reset(idx)
        self.cal_ang.reset(idx)
        self.hist[idx] = 0.0

        R = quat_to_rot(s["q"][idx])
        T_act = (self.eta[idx] * s["f"][idx]).sum(1)
        drag_acc = -self.quad.drag * (s["v"][idx] - self.wind_mean[idx]) / self.mass[idx, None]
        sf = R[:, :, 2] * (T_act / self.mass[idx])[:, None] + drag_acc
        self.truth["c"][idx] = self.eta[idx].mean(1) / self.mass[idx]
        self.truth["d"][idx] = drag_acc
        self.truth["tau"][idx] = ((self.eta[idx] - 1.0) * s["f"][idx]) @ self.quad.torque_map.T
        self._measure(idx, sf)
        self.w_meas_prev[idx] = self.meas["w"][idx]
        self.dist[idx] = np.linalg.norm(s["p"][idx] - self.goal[idx], axis=1)

    # ------------------------------------------------------------ sensors
    def _measure(self, idx, sf_true):
        k = sf_true.shape[0]
        s, sn, rng = self.state, self.sens, self.rng
        self.meas["p"][idx] = s["p"][idx] + rng.normal(0, sn["pos_noise"], (k, 3))
        self.meas["v"][idx] = s["v"][idx] + rng.normal(0, sn["vel_noise"], (k, 3))
        dq = rotvec_to_quat(rng.normal(0, sn["att_noise"], (k, 3)))
        self.meas["R"][idx] = quat_to_rot(quat_mul(s["q"][idx], dq))
        self.meas["w"][idx] = s["w"][idx] + rng.normal(0, sn["gyro_noise"], (k, 3))
        self.meas["sf"][idx] = sf_true + rng.normal(0, sn["accel_noise"], (k, 3))

    # ------------------------------------------------------------ step
    def step(self, action):
        n, quad, s = self.n, self.quad, self.state
        a = np.clip(np.asarray(action, dtype=float).reshape(n, 4), -1.0, 1.0)
        frozen = self.finished.copy()
        if frozen.any():
            saved = {k: v[frozen].copy() for k, v in s.items()}

        # 1. uncertainty arrives
        ev = self.steps == self.event_step
        if ev.any():
            self.mass[ev] = self.post["mass"][ev]
            self.eta[ev] = self.post["eta"][ev]
            self.wind_mean[ev] = self.post["wind"][ev]
            self.gust_std[ev] = self.post["gust"][ev]

        # 2. gusts
        alpha = self.dt / GUST_TIME_CONSTANT
        self.gust += (-alpha * self.gust + np.sqrt(2 * alpha) * self.gust_std[:, None]
                      * GUST_AXIS_SCALE * self.rng.normal(size=(n, 3)))
        wind = self.wind_mean + self.gust

        # 3. physics, with the action held for `substeps` steps
        T_cmd = quad.thrust_from_action(a[:, 0])
        w_cmd = quad.rates_from_action(a[:, 1:])
        acc = None
        for _ in range(self.substeps):
            f_cmd = quad.motor_commands(T_cmd, w_cmd, s["w"])
            out = quad.substep(s, f_cmd, self.mass, self.eta, wind, self.sub_dt)
            if acc is None:
                acc = {k: v.copy() for k, v in out.items()}
            else:
                for k, v in out.items():
                    acc[k] += v
        for k in acc:
            acc[k] /= self.substeps
        if frozen.any():                       # finished drones (evaluation mode) stay put
            for k in s:
                s[k][frozen] = saved[k]
        active = ~frozen

        # 4. sensors
        self._measure(slice(None), acc["sf"])
        m = self.meas

        # 5. self-calibration
        Phi_lin = np.zeros((n, 3, 4))
        Phi_lin[:, :, 0] = acc["thrust_dir_model"]
        Phi_lin[:, :, 1:] = np.eye(3)
        y_lin = m["sf"]
        y_ang = quad.J * (m["w"] - self.w_meas_prev) / self.dt + acc["gyro"] - acc["tau_model"]
        Phi_ang = np.tile(np.eye(3), (n, 1, 1))
        self.cal_lin.update(Phi_lin, y_lin, active)
        self.cal_ang.update(Phi_ang, y_ang, active)
        self.w_meas_prev = m["w"].copy()
        self.cal_inputs = {"Phi_lin": Phi_lin, "y_lin": y_lin.copy(), "y_ang": y_ang}
        safe_T = np.maximum(acc["T_model"], 1e-3)
        c_true = np.where(acc["T_model"] > 0.5, acc["T_act"] / safe_T, self.eta.mean(1)) / self.mass
        self.truth = {"c": c_true, "d": acc["drag_acc"], "tau": acc["tau_fault"]}

        # 6. reward and termination (from the true state)
        rw = self.rw
        e = s["p"] - self.goal
        dist = np.linalg.norm(e, axis=1)
        R_true = quat_to_rot(s["q"])
        yaw = yaw_of(R_true)
        crash = (s["p"][:, 2] < 0.0) | (R_true[:, 2, 2] < 0.0) | (dist > self.lost_distance)
        r = (rw["alive"]
             + rw["precision"] * np.exp(-dist / rw["precision_scale"])
             - rw["distance"] * np.minimum(dist, rw["distance_cap"])
             - rw["velocity"] * np.linalg.norm(s["v"], axis=1)
             - rw["body_rate"] * np.linalg.norm(s["w"], axis=1)
             - rw["action_change"] * np.sum((a - self.prev_action) ** 2, axis=1)
             - rw["yaw"] * np.abs(yaw))
        r = np.where(crash, r + rw["crash"], r)
        self.steps += active
        terminated = crash & active
        truncated = (self.steps >= self.max_steps) & ~crash & active
        r = np.where(active, r, 0.0)
        self.dist = np.where(active, dist, self.dist)
        self.ep_return += r

        frame = np.concatenate([m["v"] / 3.0, m["w"] / 5.0, m["sf"] / 10.0, a], axis=1)
        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = frame
        self.prev_action = a

        obs = self._obs()
        info = {}
        done = terminated | truncated
        if done.any():
            idx = np.flatnonzero(done)
            info["episodes"] = [{"return": float(self.ep_return[i]), "length": int(self.steps[i]),
                                 "crashed": bool(terminated[i]), "final_dist": float(dist[i]),
                                 "had_event": bool(0 <= self.event_step[i] < self.steps[i])}
                                for i in idx]
            if self.auto_reset:
                info["final_obs"] = obs[idx].copy()
                info["done_idx"] = idx
                self._reset_idx(idx)
                obs[idx] = self._obs(idx)
            else:
                self.finished |= done
        return obs, r, terminated, truncated, info

    # ------------------------------------------------------------ observation
    def calib_features(self, idx=slice(None)):
        th = self.cal_lin.theta[idx]
        return np.concatenate([
            ((th[:, 0] / self.c0 - 1.0) / 0.5)[:, None],
            th[:, 1:4] / 5.0,
            self.cal_ang.theta[idx] / 0.1,
            self.cal_lin.relative_std[idx],
            self.cal_ang.relative_std[idx],
        ], axis=1)

    def oracle_features(self, idx=slice(None)):
        t = self.truth
        return np.concatenate([
            ((t["c"][idx] / self.c0 - 1.0) / 0.5)[:, None],
            t["d"][idx] / 5.0,
            t["tau"][idx] / 0.1,
        ], axis=1)

    def _obs(self, idx=slice(None)):
        m = self.meas
        e = m["p"][idx] - self.goal[idx]
        k = e.shape[0]
        parts = [
            np.clip(e / 2.0, -3, 3),
            np.clip(m["v"][idx] / 3.0, -3, 3),
            np.tanh(e / 0.3),
            np.tanh(m["v"][idx] / 0.5),
            m["R"][idx].reshape(k, 9),
            np.clip(m["w"][idx] / 5.0, -3, 3),
            self.prev_action[idx],
        ]
        if self.obs_mode == "calib":
            parts.append(self.calib_features(idx))
        elif self.obs_mode == "oracle":
            parts.append(self.oracle_features(idx))
        elif self.obs_mode == "history":
            parts.append(self.hist[idx].reshape(k, -1))
        return np.concatenate(parts, axis=1).astype(np.float32)
