"""What can go wrong, and when.

A sampler draws, for k drones at once: a start state, a goal, the drone's true physical
parameters at take-off, and an optional mid-flight *event* at which the parameters jump to
new values. That jump is "uncertainty arriving": the controller is not told about it.

TrainingSampler  the domain-randomised distribution used for training (config `uncertainty`)
EvalScenario     fixed, named test scenarios with one event at t = 5 s (held out by seed)
"""
import numpy as np

from .dynamics import quat_mul, rotvec_to_quat

# Test scenarios. Each changes the drone at t = event_time; "severe" goes beyond anything
# seen in training (mass, motor damage and wind all outside the training ranges).
SCENARIOS = {
    "nominal": dict(payload=1.0, motor_eff=1.0, wind=0.0, gust=0.0,
                    text="no change: nominal drone in still air"),
    "payload": dict(payload=1.45, motor_eff=1.0, wind=0.0, gust=0.0,
                    text="payload picked up: mass +45 %"),
    "motor_fault": dict(payload=1.0, motor_eff=0.65, wind=0.0, gust=0.0,
                        text="one motor loses 35 % of its thrust"),
    "wind_gust": dict(payload=1.0, motor_eff=1.0, wind=5.0, gust=0.5,
                      text="wind rises from calm to 5 m/s, gusting"),
    "combined": dict(payload=1.45, motor_eff=0.65, wind=5.0, gust=0.5,
                     text="all three at once"),
    "severe": dict(payload=1.55, motor_eff=0.55, wind=7.0, gust=0.8,
                   text="beyond the training range: mass +55 %, motor -45 %, wind 7 m/s"),
}


def _unit_horizontal(rng, k):
    a = rng.uniform(0, 2 * np.pi, k)
    return np.stack([np.cos(a), np.sin(a), np.zeros(k)], axis=1)


def _unit_sphere(rng, k):
    v = rng.normal(size=(k, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _attitude(rng, k, max_tilt, max_yaw):
    yaw = rng.uniform(-max_yaw, max_yaw, k)
    q_yaw = rotvec_to_quat(np.stack([np.zeros(k), np.zeros(k), yaw], axis=1))
    tilt_axis = _unit_horizontal(rng, k)
    q_tilt = rotvec_to_quat(tilt_axis * rng.uniform(0, max_tilt, k)[:, None])
    return quat_mul(q_tilt, q_yaw)


class TrainingSampler:
    """Domain randomisation: random drone, random wind, and usually a random mid-flight change."""

    name = "training"

    def __init__(self, cfg):
        self.u = cfg["uncertainty"]
        self.task = cfg["task"]
        self.mass_nom = float(cfg["drone"]["mass"])
        self.dt = float(cfg["sim"]["dt"])
        self.max_steps = int(round(cfg["sim"]["episode_seconds"] / self.dt))

    def sample(self, rng, k):
        t, u = self.task, self.u
        arena = np.asarray(t["arena"], dtype=float)
        goal = rng.uniform(arena[:, 0], arena[:, 1], (k, 3))
        start = goal + _unit_sphere(rng, k) * rng.uniform(0, t["max_start_distance"], k)[:, None]
        start = np.clip(start, arena[:, 0] - [0.5, 0.5, 0.5], arena[:, 1] + [0.5, 0.5, 0.5])
        out = {
            "start_pos": start,
            "goal": goal,
            "start_quat": _attitude(rng, k, t["max_start_tilt"], 0.5),
            "start_vel": _unit_sphere(rng, k) * rng.uniform(0, t["max_start_speed"], k)[:, None],
            "start_rate": rng.uniform(-t["max_start_rate"], t["max_start_rate"], (k, 3)),
        }
        if not u["enabled"]:
            out.update(mass0=np.full(k, self.mass_nom), eta0=np.ones((k, 4)),
                       wind0=np.zeros((k, 3)), gust0=np.zeros(k), event_step=np.full(k, -1))
            out.update(mass1=out["mass0"].copy(), eta1=out["eta0"].copy(),
                       wind1=out["wind0"].copy(), gust1=out["gust0"].copy())
            return out

        mass0 = rng.uniform(*u["mass_range"], k)
        eta0 = np.ones((k, 4)) * rng.uniform(*u["battery_range"], k)[:, None]
        wind0 = _unit_horizontal(rng, k) * rng.uniform(*u["wind_speed_range"], k)[:, None]
        gust0 = rng.uniform(*u["gust_std_range"], k)

        has = rng.random(k) < u["event_prob"]
        times = rng.uniform(*u["event_time_range"], k)
        event_step = np.where(has, np.round(times / self.dt).astype(int), -1)

        comp = rng.random((k, 4)) < u["component_prob"]       # payload, motor, wind, battery
        none = ~comp.any(1)
        comp[none, rng.integers(0, 4, none.sum())] = True     # every event changes something
        comp &= has[:, None]

        mass1 = np.where(comp[:, 0],
                         np.clip(mass0 * rng.uniform(*u["payload_factor_range"], k), *u["mass_limits"]),
                         mass0)
        eta1 = eta0.copy()
        rows = np.flatnonzero(comp[:, 1])
        motors = rng.integers(0, 4, rows.size)
        eta1[rows, motors] *= rng.uniform(*u["motor_efficiency_range"], rows.size)
        eta1[comp[:, 3]] *= rng.uniform(*u["battery_sag_range"], comp[:, 3].sum())[:, None]
        wind1 = np.where(comp[:, 2:3],
                         _unit_horizontal(rng, k) * rng.uniform(*u["event_wind_speed_range"], k)[:, None],
                         wind0)
        gust1 = np.where(comp[:, 2], rng.uniform(*u["event_gust_std_range"], k), gust0)
        out.update(mass0=mass0, eta0=eta0, wind0=wind0, gust0=gust0, event_step=event_step,
                   mass1=mass1, eta1=eta1, wind1=wind1, gust1=gust1)
        return out


class EvalScenario:
    """A named test: fly to a goal 1 to 2 m away, hold it, and at t = 5 s something changes."""

    def __init__(self, name, cfg, event_time=5.0, horizon=13.0):
        if name not in SCENARIOS:
            raise KeyError(f"unknown scenario {name!r}; choose from {list(SCENARIOS)}")
        self.name = name
        self.spec = SCENARIOS[name]
        self.text = self.spec["text"]
        self.mass_nom = float(cfg["drone"]["mass"])
        self.dt = float(cfg["sim"]["dt"])
        self.event_time = event_time
        self.event_step = int(round(event_time / self.dt))
        self.max_steps = int(round(horizon / self.dt))

    def sample(self, rng, k):
        s = self.spec
        goal = rng.uniform([-1.5, -1.5, 1.5], [1.5, 1.5, 2.5], (k, 3))
        start = goal + _unit_sphere(rng, k) * rng.uniform(1.0, 2.0, k)[:, None]
        start[:, 2] = np.clip(start[:, 2], 0.6, 3.4)
        mass0 = np.full(k, self.mass_nom)
        eta0 = np.ones((k, 4))
        eta1 = eta0.copy()
        eta1[np.arange(k), rng.integers(0, 4, k)] = s["motor_eff"]
        return {
            "start_pos": start,
            "goal": goal,
            "start_quat": _attitude(rng, k, 0.05, 0.3),
            "start_vel": np.zeros((k, 3)),
            "start_rate": np.zeros((k, 3)),
            "mass0": mass0, "eta0": eta0, "wind0": np.zeros((k, 3)), "gust0": np.zeros(k),
            "event_step": np.full(k, self.event_step),
            "mass1": mass0 * s["payload"], "eta1": eta1,
            "wind1": _unit_horizontal(rng, k) * s["wind"], "gust1": np.full(k, s["gust"]),
        }
