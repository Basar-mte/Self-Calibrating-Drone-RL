"""Quadrotor physics, batched over many drones with NumPy.

State of each drone
    p  position (world frame, z up), m
    v  velocity (world frame), m/s
    q  attitude quaternion (w, x, y, z), body to world
    w  body angular rate, rad/s
    f  motor thrusts the flight controller believes it is producing (after motor lag), N

What the controller commands (the "CTBR" interface used by most learned drone controllers):
collective thrust T (N) and body rates w_cmd (rad/s). A proportional body-rate loop turns
those into torques, and the mixer turns thrust and torques into four motor commands.
All of this runs at the physics rate.

What can be wrong with the drone (the uncertainty)
    mass  a payload changes it
    eta   per-motor thrust efficiency, 1 = healthy (a damaged propeller or a weak battery lowers it)
    wind  the air moves; drag pushes the drone towards the wind velocity
The controllers are designed for the nominal values and never see the true ones,
except the "oracle" agent that is used as an upper bound.
"""
import numpy as np

GRAVITY = 9.81
G_VEC = np.array([0.0, 0.0, -GRAVITY])


# ---------------------------------------------------------------- rotations
def quat_to_rot(q):
    """(N, 4) quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices (body to world)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_mul(a, b):
    """Hamilton product of two (N, 4) quaternion arrays."""
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def quat_integrate(q, w_body, dt):
    """One explicit step of q_dot = 0.5 * q (x) (0, w), then renormalise."""
    qw, qx, qy, qz = q.T
    wx, wy, wz = w_body.T
    dq = 0.5 * np.stack([
        -qx * wx - qy * wy - qz * wz,
        qw * wx + qy * wz - qz * wy,
        qw * wy - qx * wz + qz * wx,
        qw * wz + qx * wy - qy * wx,
    ], axis=1)
    q = q + dq * dt
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rotvec_to_quat(rotvec):
    """(N, 3) rotation vectors (axis times angle) -> (N, 4) quaternions."""
    angle = np.linalg.norm(rotvec, axis=1, keepdims=True)
    safe = np.maximum(angle, 1e-12)
    axis = np.where(angle > 1e-12, rotvec / safe, 0.0)
    return np.concatenate([np.cos(0.5 * angle), axis * np.sin(0.5 * angle)], axis=1)


def yaw_of(R):
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def vee(M):
    """Skew-symmetric (N, 3, 3) -> (N, 3)."""
    return np.stack([M[:, 2, 1], M[:, 0, 2], M[:, 1, 0]], axis=1)


# ---------------------------------------------------------------- the vehicle
class Quadrotor:
    """Constants of the airframe plus the flight-controller pieces that sit below the agent."""

    def __init__(self, cfg):
        self.mass = float(cfg["mass"])
        self.J = np.asarray(cfg["inertia"], dtype=float)
        self.f_max = float(cfg["max_motor_thrust"])
        self.k_m = float(cfg["yaw_torque_coeff"])
        self.tau_m = float(cfg["motor_time_constant"])
        self.drag = float(cfg["linear_drag"])
        self.rate_gains = np.asarray(cfg["rate_gains"], dtype=float)
        self.max_rate = np.asarray(cfg["max_body_rate"], dtype=float)
        self.thrust_ratio = float(cfg["max_thrust_ratio"])
        self.hover_thrust = self.mass * GRAVITY

        # X configuration. Motors 0 and 2 spin one way, 1 and 3 the other.
        d = float(cfg["arm_length"]) / np.sqrt(2.0)
        self.motor_xy = np.array([[d, d], [-d, d], [-d, -d], [d, -d]])
        spin = np.array([1.0, -1.0, 1.0, -1.0])
        x, y = self.motor_xy[:, 0], self.motor_xy[:, 1]
        # torque of an upward force F at (x, y) is r x F = (y F, -x F, 0); plus the reaction torque
        self.torque_map = np.stack([y, -x, spin * self.k_m])          # (3, 4)
        self.alloc = np.vstack([np.ones(4), self.torque_map])        # (4, 4): f -> (T, tau)
        self.alloc_inv = np.linalg.inv(self.alloc)

    # --- action <-> physical command
    def thrust_from_action(self, a0):
        """a0 = 0 is nominal hover thrust, +1 is max_thrust_ratio times hover, -2/3 or below is zero."""
        T = self.hover_thrust * (1.0 + (self.thrust_ratio - 1.0) * a0)
        return np.clip(T, 0.0, self.thrust_ratio * self.hover_thrust)

    def action_from_thrust(self, T):
        return (T / self.hover_thrust - 1.0) / (self.thrust_ratio - 1.0)

    def rates_from_action(self, a):
        return a * self.max_rate

    # --- flight controller below the agent
    def motor_commands(self, T_cmd, w_cmd, w):
        """Proportional rate loop plus gyroscopic feed-forward, then the mixer, then saturation."""
        Jw = self.J * w
        tau = self.J * (self.rate_gains * (w_cmd - w)) + np.cross(w, Jw)
        u = np.concatenate([T_cmd[:, None], tau], axis=1)
        return np.clip(u @ self.alloc_inv.T, 0.0, self.f_max)

    # --- physics
    def substep(self, s, f_cmd, mass, eta, wind, dt):
        """Advance the batched state dict `s` by dt (semi-implicit Euler).

        Returns the quantities the calibrator and the oracle need for this substep.
        """
        s["f"] += (f_cmd - s["f"]) * (dt / self.tau_m)     # motor lag
        f_act = eta * s["f"]                               # what the motors really produce
        T_act = f_act.sum(1)
        T_model = s["f"].sum(1)                            # what the controller thinks they produce
        tau_act = f_act @ self.torque_map.T
        tau_model = s["f"] @ self.torque_map.T

        R = quat_to_rot(s["q"])
        b3 = R[:, :, 2]
        F_drag = -self.drag * (s["v"] - wind)
        sf = (b3 * T_act[:, None] + F_drag) / mass[:, None]  # specific force: what an accelerometer reads
        s["v"] += (sf + G_VEC) * dt
        s["p"] += s["v"] * dt

        w = s["w"]
        gyro = np.cross(w, self.J * w)
        s["w"] = w + (tau_act - gyro) / self.J * dt
        s["q"] = quat_integrate(s["q"], s["w"], dt)
        return {
            "sf": sf,
            "thrust_dir_model": b3 * T_model[:, None],   # regressor of the thrust gain
            "tau_model": tau_model,
            "gyro": gyro,
            "T_act": T_act,
            "T_model": T_model,
            "drag_acc": F_drag / mass[:, None],
            "tau_fault": tau_act - tau_model,
        }
