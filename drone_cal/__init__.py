"""Self-calibrating drone: a quadrotor that re-identifies itself when uncertainty arrives.

Modules
-------
dynamics      quadrotor physics, motor mixing and the inner body-rate loop
calibrator    online Kalman estimator with a CUSUM change detector
scenarios     what can go wrong, and when: the training distribution and the test scenarios
env           the batched simulation the agents and controllers fly in
sac           Soft Actor-Critic (networks, replay buffer, agent)
pid           a classical cascaded controller, with or without the calibrator
evaluation    rollouts and metrics
viz           figures and animations
utils         config loading, seeding, logging
"""
__version__ = "1.0.0"
