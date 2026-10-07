# 1. The ideas, in plain words

This page explains every idea the project uses, from "how does a drone fly" to "how does
it notice that something has changed". No equations are needed to follow it; the few that
appear are there for readers who want them.

Next: [02_setup.md](02_setup.md).

---

## 1.1 The problem in one paragraph

A controller is designed for a drone it believes it knows: a certain mass, four healthy
motors, still air. Then, in the middle of a flight, the drone picks up a parcel, a
propeller chips, or the wind rises. Nobody tells the controller. It keeps pushing as if
nothing happened, and the drone sags, tilts and drifts. **This project gives the drone a
way to notice that its model of itself has stopped fitting, re-measure itself within a
fraction of a second, and keep flying.** That is what "self-calibrating" means here.

## 1.2 How a quadrotor flies

A quadrotor has four propellers, each pushing straight up along the body's own "up" axis.

* **Going up and down.** All four push harder or softer together. Their sum is the
  *collective thrust*.
* **Tilting.** If the two right motors push harder than the two left ones, the drone rolls
  left. Pitch works the same way front to back.
* **Moving sideways.** A drone cannot push sideways. It tilts, and part of its thrust
  then points sideways. To move, it must first rotate.
* **Turning on the spot (yaw).** Half the propellers spin clockwise, half
  anticlockwise. Speeding up one pair twists the frame the other way.

The simulator in [drone_cal/dynamics.py](../drone_cal/dynamics.py) computes exactly this:
four motor forces give one total force and three torques; force changes velocity,
torque changes rotation speed. It runs at 200 steps per second.

### The layers of a flight controller

Real flight software is built in layers, and so is this project:

```
   the agent or the PID            50 times a second
        |  collective thrust + desired rotation speeds (roll, pitch, yaw rates)
        v
   body-rate loop                   200 times a second
        |  torques needed to reach those rotation speeds
        v
   mixer                            which motor does what
        |  four motor commands
        v
   motors (with a 30 ms lag) -> the physical drone
```

The top layer's output is called **CTBR** (collective thrust and body rates). It is what
most learned drone controllers output, including the ones that have beaten human racing
pilots. The rate loop below it is *proportional only*: it pushes in proportion to how far
the rotation speed is from the target. That matters later.

## 1.3 What "uncertainty" means here

Three kinds of things can be wrong, and the controller is never told about them:

| what changes | what you would see | how this project models it |
|---|---|---|
| **mass** (a payload) | the drone sinks: the same thrust lifts less | true mass jumps, e.g. +45 % |
| **motor health** (a damaged propeller, a weak battery) | the drone tilts towards the weak corner, sinks a little, and yaws slowly | one motor's efficiency drops, e.g. to 65 % |
| **wind** | the drone drifts downwind | the air moves at up to 7 m/s, with gusts |

Each test scenario changes one or more of these at **t = 5 s**, while the drone is
hovering at its goal. That moment is "uncertainty arriving". Training uses random versions
of the same changes, at random times (see [04_config_reference.md](04_config_reference.md)).

## 1.4 Why a fixed controller struggles

A controller designed for the nominal drone makes three wrong assumptions after the change:

* **"This much thrust holds me up."** With a payload or a weak motor it does not, so the
  drone sinks until an error builds up big enough to correct.
* **"No force pushes me sideways."** The wind does, so the drone drifts until the
  position error builds up.
* **"The rate loop delivers the rotation I ask for."** With a weak motor, the
  proportional rate loop settles at the wrong rotation speed. A weak corner produces a
  constant torque, and a proportional loop only pushes back once it sees an error. So the
  drone keeps tilting a little more than asked.

A controller with an *integrator* (the I in PID) fixes all three eventually: it adds up
the error over time and pushes harder the longer the error lasts. But "eventually" is the
problem. The error has to exist, and grow, before the integrator reacts. In the hardest
test the plain PID needs about 7 seconds and drifts more than a metre first.

## 1.5 Self-calibration: measuring yourself while flying

The drone already carries the instruments it needs:

* an **accelerometer**, which measures how hard the drone is actually being pushed, and
* a **gyroscope**, which measures how fast it is actually rotating.

It also knows what it *asked* the motors for. Comparing the two answers "am I getting the
push and the torque I expect?". [drone_cal/calibrator.py](../drone_cal/calibrator.py) turns
that comparison into numbers, using two small models of the drone:

**Translation model.** *Measured push = c × (commanded thrust, along the body's up axis) + d*

* **c** is the *thrust gain*: how much acceleration one newton of commanded thrust really
  produces. For the nominal 1 kg drone it is 1. A heavier drone or a weaker motor makes it
  smaller. Only this combination matters for control, which is lucky, because mass and
  motor strength cannot be told apart from these measurements alone.
* **d** is the *disturbance*: any extra push, such as wind drag.

**Rotation model.** *Measured rotational acceleration × inertia − expected torque = τ*

* **τ** is the *fault torque*: the twist the motors produce but were not asked for. A weak
  motor makes it non-zero.

These seven numbers (c, three components of d, three of τ) are what the drone calibrates.

### The Kalman filter: a careful running average

Every 20 ms there is a new measurement, and every measurement is noisy. A **Kalman
filter** combines them into a best estimate. It also tracks its own **uncertainty**: how
sure it is of each number.

In plain terms, every step it:

1. predicts the measurement from its current estimates;
2. looks at the prediction error (the **innovation**);
3. moves its estimate part of the way towards what would have explained the
   measurement. "Part of the way" is large when it is unsure and small when it is sure.
4. becomes a little more sure, because it has seen one more measurement.

It also assumes the true values can drift slowly, so its uncertainty never shrinks all the
way to zero. That drift setting is the dilemma at the heart of the project:

* **trust the estimate (slow drift):** smooth, accurate numbers in steady flight, but it
  takes seconds to believe a sudden change, and can settle on a wrong answer;
* **forget fast (fast drift):** quick to follow a change, but noisy all the time.

The study in [calibration_study.py](../calibration_study.py) shows both failure modes.

### The change detector: noticing that the model stopped fitting

The way out is to be slow *by default* and fast *only when needed*. To do that, the drone
must know when "needed" is. The prediction errors tell it:

* While the model is right, the errors are just noise, of a size the filter can predict.
  The **normalised innovation** (error squared, divided by its expected size) averages
  about 3 for three measured numbers.
* When the drone changes, the errors become systematically too big.

A **CUSUM** (cumulative sum) detector adds up how much each normalised error exceeds what
is expected. Normal noise keeps the running total near zero; a real change makes it climb.
When it crosses a threshold, the detector raises an **alarm**:

> "My model of myself no longer fits. Uncertainty has arrived."

On an alarm the filter **resets its uncertainty to its take-off level**. Being unsure
again, it takes big steps, and the estimates snap to the new values within a few tenths
of a second. Then they settle back into smooth, slow tracking. In nominal flight it never
fires: the study measured zero false alarms over 100 flights.

## 1.6 Reinforcement learning, briefly

**Reinforcement learning (RL)** learns a controller by trial and error. The agent observes
the drone, chooses an action, and receives a **reward** that says how good the moment was.
Over millions of steps it learns which actions lead to high total reward.

The reward here, every 20 ms, is roughly:

* **+1** for still flying, plus up to **+1** more for being very close to the goal;
* minus small penalties for distance, speed, spinning, jerky commands and facing the wrong way;
* **−10** once, for a crash (hitting the ground, flipping over, or drifting 6 m away).

### SAC: the learning algorithm

**Soft Actor-Critic (SAC)** suits continuous actions like "thrust = 0.37, roll rate = −0.12".
It trains three networks together:

* the **actor** (the policy) maps an observation to a *range* of actions, a Gaussian
  bell curve. It samples from that range while learning and takes the centre when flying
  for real;
* two **critics** each estimate how much future reward follows an action. Using the
  smaller of the two estimates guards against the critics fooling themselves with
  over-optimistic numbers.

The "soft" part: SAC is also rewarded for keeping some randomness (entropy) in its
actions, which keeps it exploring. A temperature α, tuned automatically, sets the balance.
Every line is in [drone_cal/sac.py](../drone_cal/sac.py), with no RL library.

## 1.7 How the two halves fit together

The calibrator and the learned controller are separate on purpose:

```
   sensors --> calibrator --> estimates (c, d, τ) and their uncertainty
      |                              |
      v                              v
   observation  ------------------>  agent (SAC)  --> thrust + body rates --> drone
```

The agent is trained on randomised drones, with random changes at random times, so it
learns *how to use* the estimates: lean into the wind by d, push harder when c drops,
counter-steer by τ, and be careful while the uncertainty is high.

To find out whether the calibrator actually helps, the same recipe is trained with
different observations:

| agent | sees | question it answers |
|---|---|---|
| **SAC + calibrator** | the drone's state + the calibrator's estimates and uncertainty | the method |
| **SAC, blind** | the drone's state only, but trained on the same random drones | can a policy just be robust, without calibrating? |
| **SAC + history** | the drone's state + its last 10 sensor readings and actions | can a network work out the change by itself from raw history? |
| **SAC + true parameters** | the drone's state + the exact values the calibrator estimates | the upper bound: perfect, instant calibration |
| **SAC, nominal training** | the drone's state, trained only on the perfect drone | what happens with no preparation for uncertainty at all |

And two classical controllers, flown on exactly the same drones:

| controller | what it is |
|---|---|
| **PID** | the textbook cascaded controller, tuned for the nominal drone, with an integrator |
| **PID + calibrator** | the same, using the calibrator's estimates as feed-forward |

The results, and what they do and do not show, are in the [README](../README.md#results)
and on the [results figures](../results/figures/).
