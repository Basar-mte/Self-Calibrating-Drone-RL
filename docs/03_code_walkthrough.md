# 3. Code walkthrough

What each file does, in the order worth reading them. Line counts are approximate.

Previous: [02_setup.md](02_setup.md). Next: [04_config_reference.md](04_config_reference.md).

---

## 3.1 Map of the package

```
drone_cal/
  dynamics.py     the physics: rotations, motors, mixer, rate loop, rigid body     (~170 lines)
  calibrator.py   Kalman estimator + CUSUM change detector                         (~130)
  scenarios.py    what goes wrong and when: training distribution, test scenarios  (~150)
  env.py          the batched world the controllers fly in                         (~300)
  pid.py          the classical cascaded controller                                (~80)
  sac.py          Soft Actor-Critic: networks, replay buffer, agent, CUDA graph    (~240)
  evaluation.py   rollouts and metrics                                             (~130)
  viz.py          figures and the GIF                                              (~330)
  utils.py        config, seeding, CSV logging                                     (~80)
train.py  evaluate.py  calibration_study.py  demo.py  plot_training.py  run_all.sh
tests/test_env.py
```

Everything is **batched**: every array has the number of drones as its first dimension,
so one call flies 16 drones in training, or 100 in evaluation, at almost the cost of one.

## 3.2 `dynamics.py`: the physics

**Rotations.** Attitude is a unit quaternion `q = (w, x, y, z)`. `quat_to_rot` turns it
into a rotation matrix whose third column `b3` is the body's "up" axis, the direction all
thrust points. `quat_integrate` advances it by one step of the angular velocity and
renormalises.

**The airframe** (`Quadrotor.__init__`). Motors sit at the four corners of an X, at
`arm_length / sqrt(2)` along x and y. An upward force `F` at `(x, y)` produces torque
`(y F, -x F, 0)`. Each propeller's drag also twists the frame about z by
`± yaw_torque_coeff × F`, depending on its spin direction. Stacking the four motors gives
the 4 × 4 **allocation matrix** `alloc`, which maps motor thrusts to (total thrust, three
torques). Its inverse is the **mixer**.

**The flight controller below the agent** (`motor_commands`):

```
tau = J * rate_gains * (w_cmd - w)  +  w x (J w)      proportional rate loop + gyroscopic term
f   = clip(alloc_inv @ [T, tau], 0, f_max)            mixer, then motor saturation
```

**One physics step** (`substep`, 200 Hz):

```
f      += (f_cmd - f) * dt / tau_m        motor lag (first order, 30 ms)
f_act   = eta * f                         what the motors really push: eta < 1 is a weak motor
sf      = (b3 * sum(f_act) + drag) / m    specific force (what the accelerometer reads)
v      += (sf + g) * dt;  p += v * dt     translation (semi-implicit Euler)
w      += (tau_act - w x Jw) / J * dt     rotation
```

The function also returns what the calibrator and the oracle need: the thrust *the
controller believes* it commanded (`T_model`, from `f`, not `f_act`), the torque it
believes it commanded, and the true fault torque and drag for the oracle.

## 3.3 `calibrator.py`: self-calibration

`KalmanCalibrator.update(Phi, y)` is one Kalman step for every drone at once. The
docstring at the top of the file explains the models; here is the step itself:

```
P  = P + Q                       predict: parameters may drift (random walk)
e  = y - Phi theta               innovation
S  = Phi P Phi^T + R             its expected covariance
K  = P Phi^T S^-1                gain: how far to move
theta += K e                     update the estimate
P  = (I - K Phi) P (...)^T + K R K^T     Joseph form: stays symmetric and positive
nis = e^T S^-1 e                 normalised innovation squared
g   = max(0, g + nis - m - k)    CUSUM
if g > h:  P += P0, g = 0        ALARM: reopen the estimate
```

`make_calibrators` builds the two instances the environment uses:

| | parameters | measurement | regressor `Phi` |
|---|---|---|---|
| translation | `c, d_x, d_y, d_z` | accelerometer (world frame) | `[T_model * b3, I3]` |
| rotation | `tau_x, tau_y, tau_z` | `J * dw/dt + w x Jw - tau_model`, from the gyro | `I3` |

The vertical disturbance `d_z` gets a small prior and a small drift. Without that, a
drone hovering at constant thrust could not tell "my thrust gain fell" from "something
pushes me down": both look the same in the accelerometer. The prior encodes the physical
fact that wind is mostly horizontal, so a vertical misfit is blamed on the thrust gain.

## 3.4 `scenarios.py`: what goes wrong, and when

A sampler returns, for `k` drones: start state, goal, the true parameters at take-off
(`mass0, eta0, wind0, gust0`), the event step, and the parameters after the event
(`mass1, eta1, wind1, gust1`).

* `TrainingSampler`: random drones (mass, battery, wind) and, with 80 % probability, a
  random event between 1.5 and 8 s. The event changes each of payload, one motor, wind
  and battery with 50 % probability, and always changes at least one.
* `EvalScenario`: the six named tests in `SCENARIOS`. The drone starts 1 to 2 m from its
  goal, healthy and in still air. The change comes at t = 5 s, and the flight lasts 13 s.

## 3.5 `env.py`: the world

`DroneEnv.step(action)` in order:

1. **Events.** Drones whose `event_step` has come get their new mass, motor efficiencies and wind.
2. **Gusts.** An Ornstein-Uhlenbeck process (random, but pulled back to zero with a 1 s
   time constant), weaker vertically.
3. **Physics.** The action is turned into thrust and rate commands and held for 4
   substeps. Each substep runs rate loop, mixer and physics. Calibrator inputs are
   averaged over the substeps.
4. **Sensors.** Noise is added to position, velocity, attitude, gyro and accelerometer.
5. **Calibration.** Both Kalman filters update.
6. **Reward and termination**, from the *true* state (the agent sees only the noisy one).
   A crash is hitting the ground, turning upside down, or drifting 6 m from the goal.

**The observation** (`_obs`). Every mode starts with the same 28 numbers:

| index | content | scaling |
|---|---|---|
| 0 to 2 | position error (measured position minus goal) | / 2 m, clipped to ±3 |
| 3 to 5 | velocity | / 3 m/s, clipped |
| 6 to 8 | position error, fine scale | tanh(error / 0.3 m) |
| 9 to 11 | velocity, fine scale | tanh(v / 0.5 m/s) |
| 12 to 20 | rotation matrix, row by row | none needed |
| 21 to 23 | body rates | / 5 rad/s, clipped |
| 24 to 27 | previous action | already in [-1, 1] |

Then, depending on `observation.mode`:

| mode | extra numbers |
|---|---|
| `calib` (+14) | `(c/c0 - 1)/0.5`, `d / 5`, `tau / 0.1`, then each parameter's uncertainty as a fraction of its take-off uncertainty (7 numbers) |
| `oracle` (+7) | the same first 7 numbers, computed from the truth |
| `history` (+130) | the last 10 frames of (velocity / 3, body rates / 5, accelerometer / 10, action) |
| `base` | nothing |

The fine-scale copies matter. With only the coarse scaling, a 5 cm error is 0.025, a
signal too small for the network to hold position precisely; an early pilot without them
hovered 0.4 to 0.6 m off target. Through tanh at 0.3 m, the same error is 0.17.

**The reward**, per 20 ms step:

```
r =  1.0                                   alive
   + 1.0 * exp(-distance / 0.3)            precision bonus
   - 0.2 * min(distance, 4)                distance
   - 0.05 * speed - 0.05 * |body rate|     smoothness
   - 0.05 * |action - previous action|^2
   - 0.1 * |yaw|
   - 10 on a crash
```

The alive term keeps every in-flight reward positive. If flying far from the goal earned
negative reward, crashing early would end the negative stream and look attractive.

**Auto-reset.** In training (`auto_reset=True`), a drone whose episode ends is reset at
once. Its real final observation goes in `info["final_obs"]`, so the replay buffer
stores the true last transition. In evaluation (`auto_reset=False`) finished drones are
frozen and the episode runs to the time limit.

## 3.6 `sac.py`: the agent

* `Actor`: MLP 42 → 256 → 256 → (mean, log std) for 4 actions; tanh squashing with the
  log-probability correction.
* `Critic`: two independent MLPs (obs + action) → 256 → 256 → 1.
* `ReplayBuffer`: 1M transitions, float32 NumPy arrays.
* `SACAgent._update_tensors`: one SAC step, exactly as listed in the file's docstring.
  It contains no `.item()` calls, so it can run inside a CUDA graph.

**Why a CUDA graph.** A SAC update launches a few hundred tiny GPU operations. Each launch
costs CPU time, and on a busy laptop that overhead dominated: 22 ms per update, against a
GPU that was mostly idle. `_capture` records the whole update once, after three warm-up
steps. Every later update copies the new batch into the recorded input tensors and calls
`graph.replay()`, a single launch: 1.75 ms. The optimisers are created with
`capturable=True` so their step counters live on the GPU. On a CPU, or with
`sac.cuda_graph: false`, the same function runs normally.

## 3.7 `train.py`: the loop

```
for every vector step (16 drones):
    action = random for the first 20k steps, then sampled from the actor
    step the env; store (s, a, r, s', crashed) in the replay buffer
        (a time-limit end is stored as not-done, so the critic bootstraps through it)
    4 SAC updates on batches of 256
    every 50k steps: fly 48 held-out drones deterministically; save best.pt by mean return
```

## 3.8 `pid.py`: the classical baseline

```
a_cmd   = -kp e - kd v - ki integral(e)              desired acceleration (limited)
F       = a_cmd + g - d_hat                          force per kg the thrust must supply
thrust  = (F . b3) / c_hat                           projected on the current up axis
R_des   = attitude with up axis along F, yaw 0
w_cmd   = -k_att * attitude_error - tau_hat / (J * rate_gains)
```

Without the calibrator, `c_hat = 1/m_nominal`, `d_hat = 0`, `tau_hat = 0`. The last term
cancels the steady rate error the proportional rate loop would otherwise show under a
fault torque. The integrator limit is set high enough (5 m s) that it can compensate
every test scenario given time, so the plain PID is slow but not crippled.

## 3.9 `evaluation.py`, `evaluate.py`, `calibration_study.py`

`run_episodes` flies a full episode on every drone and records whatever the figures
need. `episode_metrics` computes the metrics defined in its docstring and in
the [README](../README.md#results). `evaluate.py` uses scenario seed
`50,000 + index`, so every controller meets the same drones.

`calibration_study.py` separates the estimator from the controller. It records what the
calibrator was fed during PID flights, then replays the identical data through three
estimator settings.
