# 4. Every setting explained

All settings live in [configs/default.yaml](../configs/default.yaml). Any of them can be
changed from the command line without editing the file:

```bash
python train.py --run-name test --set observation.mode=base reward.crash=-20 train.total_steps=500000
```

The run folder keeps the complete settings it was trained with (`config.yaml`), so a run
can always be repeated.

Previous: [03_code_walkthrough.md](03_code_walkthrough.md). Next: [05_glossary.md](05_glossary.md).

---

## `drone`: the airframe and the flight controller below the agent

| key | default | meaning |
|---|---|---|
| `mass` | 1.0 | kg. The *nominal* mass the controllers are designed for; the true mass varies |
| `arm_length` | 0.17 | m from the centre to each motor |
| `inertia` | [0.0082, 0.0082, 0.0149] | kg m² about body x, y, z |
| `max_motor_thrust` | 8.0 | N per motor; thrust-to-weight about 3.3 at 1 kg |
| `yaw_torque_coeff` | 0.016 | m; reaction torque per newton of thrust (propeller drag) |
| `motor_time_constant` | 0.03 | s; the first-order motor lag |
| `linear_drag` | 0.25 | N per m/s of airspeed; how strongly wind pushes |
| `rate_gains` | [20, 20, 10] | 1/s; the proportional body-rate loop |
| `max_body_rate` | [5, 5, 2] | rad/s commanded when the action is ±1 |
| `max_thrust_ratio` | 2.5 | collective thrust at action +1, as a multiple of nominal hover thrust |

## `sim`

| key | default | meaning |
|---|---|---|
| `dt` | 0.02 | s; one control step (50 Hz) |
| `substeps` | 4 | physics steps per control step (200 Hz) |
| `episode_seconds` | 10.0 | training episode length (test scenarios use 13 s) |

## `sensors`: measurement noise (standard deviations)

`pos_noise` 0.01 m, `vel_noise` 0.02 m/s, `att_noise` 0.005 rad, `gyro_noise` 0.003 rad/s,
`accel_noise` 0.15 m/s². Setting all to 0 gives perfect sensors (used by one sanity check).

## `task`

| key | default | meaning |
|---|---|---|
| `arena` | x, y in [-2.5, 2.5], z in [1, 3] | box goals are drawn from, m |
| `max_start_distance` | 3.0 | m; training starts at most this far from the goal |
| `max_start_tilt` | 0.3 | rad |
| `max_start_speed` | 0.5 | m/s |
| `max_start_rate` | 0.5 | rad/s |
| `lost_distance` | 6.0 | m from the goal: counted as a crash |

## `reward`: see 3.5 for the formula

| key | default | effect of raising it |
|---|---|---|
| `alive` | 1.0 | values survival; keep it large enough that flying always beats crashing |
| `precision` | 1.0 | values being very close to the goal |
| `precision_scale` | 0.3 | m; a smaller value rewards only very tight hovering |
| `distance` | 0.2 | pulls harder towards the goal from far away |
| `distance_cap` | 4.0 | m; distances beyond this cost no more |
| `velocity` | 0.05 | calmer flight, slower approach |
| `body_rate` | 0.05 | less spinning and wobbling |
| `action_change` | 0.05 | smoother commands |
| `yaw` | 0.1 | holds the nose pointing along x |
| `crash` | -10.0 | more careful, but can make learning slower |

## `uncertainty`: the training distribution

| key | default | meaning |
|---|---|---|
| `enabled` | true | false trains on the perfect drone only (the "nominal training" agent) |
| `mass_range` | [0.85, 1.15] | kg at take-off |
| `battery_range` | [0.92, 1.0] | efficiency of all motors at take-off |
| `wind_speed_range` | [0, 3] | m/s at take-off, horizontal, random direction |
| `gust_std_range` | [0, 0.6] | m/s |
| `event_prob` | 0.8 | chance of a mid-flight change |
| `event_time_range` | [1.5, 8.0] | s |
| `component_prob` | 0.5 | chance that each kind of change is part of the event (at least one always is) |
| `payload_factor_range` | [0.8, 1.45] | mass multiplied by this (below 1 is dropping a load) |
| `mass_limits` | [0.7, 1.5] | kg; the result is clipped to this |
| `motor_efficiency_range` | [0.6, 0.95] | one random motor drops to this fraction |
| `event_wind_speed_range` | [0, 6] | m/s |
| `event_gust_std_range` | [0, 0.8] | m/s |
| `battery_sag_range` | [0.88, 1.0] | all motors multiplied by this |

The "severe" test scenario uses a mass of 1.55 kg, one motor at 55 % and 7 m/s wind. All
three are outside these ranges on purpose.

## `calibrator`

| key | default | meaning |
|---|---|---|
| `trigger` | true | the CUSUM change detector; false gives a plain Kalman filter |
| `thrust_gain_prior_std` | 0.3 | 1/kg; take-off uncertainty of c, and how much is added back on an alarm |
| `disturbance_prior_std` | [3, 3, 0.5] | m/s²; small vertically on purpose (see 3.3) |
| `torque_prior_std` | 0.15 | N m |
| `thrust_gain_drift` | 0.001 | per step; how fast c may change by itself |
| `disturbance_drift` | [0.02, 0.02, 0.002] | per step; wind changes faster than mass |
| `torque_drift` | 0.0005 | per step |
| `accel_meas_std` | 0.3 | m/s²; accelerometer noise plus model mismatch |
| `torque_meas_std` | 0.006 | N m |
| `cusum_slack` | 3.0 | how much excess misfit per step is ignored as noise |
| `cusum_threshold` | 30.0 | accumulated misfit that raises an alarm. Lower reacts faster and risks false alarms |

## `observation`

| key | default | meaning |
|---|---|---|
| `mode` | calib | `base`, `calib`, `oracle` or `history` (table in 3.5) |
| `history` | 10 | frames kept, for `history` mode |

## `sac`

| key | default | meaning |
|---|---|---|
| `hidden` | [256, 256] | hidden layer sizes of actor and critics |
| `gamma` | 0.99 | discount; about 2 s of look-ahead at 50 Hz |
| `tau` | 0.005 | target-critic averaging rate |
| `lr` | 0.0003 | Adam learning rate for all three optimisers |
| `batch_size` | 256 | transitions per update |
| `buffer_size` | 1,000,000 | replay memory |
| `init_alpha` | 0.2 | starting temperature |
| `target_entropy` | auto | -4 (minus the number of action dimensions) |
| `cuda_graph` | auto | record the update as a CUDA graph on GPU (same maths, about 10x faster) |

## `train`

| key | default | meaning |
|---|---|---|
| `total_steps` | 1,500,000 | environment steps, summed over parallel drones |
| `num_envs` | 16 | drones simulated in parallel |
| `start_steps` | 20,000 | random actions before the policy takes over |
| `learning_starts` | 10,000 | no updates before this |
| `updates_per_step` | 4 | gradient steps per vector step (one update per 4 environment steps) |
| `eval_every` | 50,000 | steps between held-out evaluations |
| `eval_episodes` | 48 | held-out flights per evaluation |

A pilot with 8 updates per step and batch 512 learned faster at first, then collapsed to
100 % crashes for a while (the temperature jumped). The default stays at 4.

## Recipes

```bash
# every agent in the README comparison (12 runs, about 3 h on a laptop GPU)
bash run_all.sh

# a stricter change detector: fewer alarms, slower to react
python evaluate.py --runs results/sac_calib_s0 --pid   # after editing calibrator.cusum_threshold

# harder training: stronger faults and wind
python train.py --run-name hard --set uncertainty.motor_efficiency_range=[0.5,0.9] uncertainty.event_wind_speed_range=[0,8]
```
