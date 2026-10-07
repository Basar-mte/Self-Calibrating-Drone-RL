# 5. Glossary

Previous: [04_config_reference.md](04_config_reference.md).

---

| term | meaning |
|---|---|
| **actor** | the policy network: maps an observation to a distribution over actions |
| **alarm** | the change detector deciding that the calibrator's model no longer fits the drone |
| **attitude** | which way the drone is pointing (roll, pitch, yaw); stored as a quaternion |
| **body rates** | how fast the drone rotates about its own axes, rad/s |
| **bootstrapping** | estimating the value of the future from the critic's own prediction, instead of waiting for the episode to end |
| **calibration** | here: estimating, during flight, the numbers that describe the drone's current physics (thrust gain, disturbance, fault torque) |
| **critic** | a network that estimates the future reward of taking an action in a state |
| **CTBR** | collective thrust and body rates: the command interface between the agent and the flight controller |
| **CUDA graph** | a recording of a sequence of GPU operations, replayed with one launch; removes per-operation overhead |
| **CUSUM** | cumulative sum: a change detector that adds up evidence of a misfit and alarms when the total crosses a threshold |
| **disturbance d** | extra acceleration the model does not explain, mostly wind drag, m/s² |
| **domain randomisation** | training on many random versions of the drone so that the policy does not depend on one exact model |
| **drift (of a parameter)** | how fast the Kalman filter assumes a parameter can change by itself, per step |
| **entropy** | how spread out the policy's action distribution is; SAC keeps it near a target |
| **episode** | one flight, from reset to the goal-holding time limit or a crash |
| **event** | the moment in an episode when the drone's true parameters jump: uncertainty arriving |
| **fault torque τ** | the torque the motors produce that the flight controller did not ask for (a weak motor), N m |
| **feed-forward** | using a known or estimated quantity to act before an error appears, instead of correcting the error after |
| **gyroscope** | sensor measuring body rates |
| **hold error** | mean distance to the goal over the last 2 s of an episode |
| **innovation** | the calibrator's prediction error: measurement minus what the current estimates predicted |
| **integrator** | the I in PID: adds up error over time, so a constant error eventually gets a constant correction |
| **Joseph form** | a numerically stable way of updating the Kalman covariance |
| **Kalman filter** | a recursive estimator that combines noisy measurements into a best estimate and tracks its uncertainty |
| **mixer** | converts collective thrust and three torques into four motor commands |
| **NIS** | normalised innovation squared: the innovation measured in units of its expected size; about m on average for m measurements when the model is right |
| **observation** | the numbers the agent receives each step |
| **oracle** | an agent given the true values of what the calibrator estimates; an upper bound, impossible on a real drone |
| **Polyak averaging** | moving the target critics a small fraction tau towards the critics after every update |
| **quaternion** | four numbers describing a rotation without the singularities of roll-pitch-yaw angles |
| **rate loop** | the inner controller that turns body-rate commands into torques; proportional only here |
| **replay buffer** | memory of past transitions that SAC samples its training batches from |
| **reparameterisation trick** | writing a random action as mean + std x noise so gradients can flow through the sampling |
| **reward** | the number returned every step that defines what good flying is |
| **SAC** | Soft Actor-Critic: an off-policy actor-critic algorithm with automatic entropy tuning |
| **settle time** | time after the event until the distance to the goal stays below 0.25 m for good |
| **specific force** | what an accelerometer measures: acceleration minus gravity |
| **temperature α** | the weight of the entropy bonus in SAC; tuned automatically |
| **thrust gain c** | acceleration produced per newton of commanded thrust; 1/mass for a healthy drone, lower with payload or damage, 1/kg |
| **truncation** | an episode ending because of the time limit, not because of a crash; the future is still bootstrapped |
| **uncertainty (of an estimate)** | the calibrator's own standard deviation for each parameter, from the Kalman covariance |
