"""A classical cascaded controller: the baseline every learned controller has to beat.

    position error --PID--> desired acceleration --geometry--> thrust and desired attitude
    attitude error --P-->   body-rate command          (the same CTBR interface as the agent)

Without calibration it assumes the nominal drone: thrust gain 1/m, no disturbance, no fault.
Its integrator still removes constant errors, but slowly, by letting the error build up first.

With calibration it uses the calibrator's estimates as feed-forward, the textbook way:
    thrust      divide the needed force by the estimated thrust gain c
    disturbance subtract the estimated disturbance acceleration d
    fault       pre-bias the rate command by the estimated fault torque, which the
                proportional rate loop would otherwise turn into a steady rate error
"""
import numpy as np

from .dynamics import G_VEC, vee


class CascadedPID:
    def __init__(self, cfg, n, use_calibration=False,
                 kp=(4.0, 4.0, 5.0), kd=(3.0, 3.0, 3.5), ki=(1.0, 1.0, 1.5),
                 k_att=8.0, k_yaw=3.0, max_tilt_deg=35.0, max_acc_xy=6.0, i_limit=5.0):
        self.name = "PID + calibrator" if use_calibration else "PID"
        self.use_cal = use_calibration
        self.n = n
        self.dt = float(cfg["sim"]["dt"])
        self.kp, self.kd, self.ki = map(lambda g: np.asarray(g, dtype=float), (kp, kd, ki))
        self.k_att = np.array([k_att, k_att, k_yaw])
        self.tan_tilt = np.tan(np.radians(max_tilt_deg))
        self.max_acc_xy = max_acc_xy
        self.i_limit = i_limit
        self.I = np.zeros((n, 3))

    def reset(self, n=None):
        if n is not None:
            self.n = n
        self.I = np.zeros((self.n, 3))

    def act(self, obs, env):
        quad, m = env.quad, env.meas
        p, v, R = m["p"], m["v"], m["R"]
        e = p - env.goal
        self.I = np.clip(self.I + e * self.dt, -self.i_limit, self.i_limit)
        a_cmd = -self.kp * e - self.kd * v - self.ki * self.I
        h = np.linalg.norm(a_cmd[:, :2], axis=1, keepdims=True)
        a_cmd[:, :2] *= np.minimum(1.0, self.max_acc_xy / np.maximum(h, 1e-9))
        a_cmd[:, 2] = np.clip(a_cmd[:, 2], -6.0, 8.0)

        if self.use_cal:
            c_hat = np.maximum(env.cal_lin.theta[:, 0], 0.2)
            d_hat = env.cal_lin.theta[:, 1:4]
            tau_hat = env.cal_ang.theta
        else:
            c_hat = np.full(self.n, 1.0 / quad.mass)
            d_hat = np.zeros((self.n, 3))
            tau_hat = np.zeros((self.n, 3))

        F = a_cmd - G_VEC - d_hat                      # specific force the thrust must provide
        F[:, 2] = np.maximum(F[:, 2], 1.0)
        h = np.linalg.norm(F[:, :2], axis=1, keepdims=True)
        lim = self.tan_tilt * F[:, 2:3]
        F[:, :2] *= np.minimum(1.0, lim / np.maximum(h, 1e-9))
        b3_des = F / np.linalg.norm(F, axis=1, keepdims=True)
        T = np.maximum(np.einsum("ni,ni->n", F, R[:, :, 2]), 0.0) / c_hat

        b1c = np.tile([1.0, 0.0, 0.0], (self.n, 1))     # hold yaw = 0
        b2 = np.cross(b3_des, b1c)
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        b1 = np.cross(b2, b3_des)
        R_des = np.stack([b1, b2, b3_des], axis=2)
        Rt = np.transpose(R, (0, 2, 1))
        Rdt = np.transpose(R_des, (0, 2, 1))
        e_R = 0.5 * vee(Rdt @ R - Rt @ R_des)
        w_cmd = -self.k_att * e_R
        # the P rate loop settles at w = w_cmd + tau_d / (J k_w): cancel that offset
        w_cmd -= tau_hat / (quad.J * quad.rate_gains)

        act = np.empty((self.n, 4))
        act[:, 0] = quad.action_from_thrust(T)
        act[:, 1:] = w_cmd / quad.max_rate
        return np.clip(act, -1.0, 1.0)
