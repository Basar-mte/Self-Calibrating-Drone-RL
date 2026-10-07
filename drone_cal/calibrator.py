"""Online self-calibration: a Kalman parameter estimator with a CUSUM change detector.

The drone keeps two small linear models of itself and fits them while it flies.

Translational model (4 unknowns), from the accelerometer:
    specific_force = c * (T_model * b3) + d
    c   thrust gain, 1/kg. Real thrust divided by commanded thrust, divided by mass.
        Nominal 1/m. A payload, a weak battery or a damaged motor all lower it,
        and only their combination matters for control.
    d   disturbance acceleration, m/s^2: wind drag and anything else unmodelled.

Rotational model (3 unknowns), from the gyro:
    J * dw/dt + w x Jw - tau_model = tau_d
    tau_d  disturbance torque, N m. A damaged motor makes one corner weaker, which
           shows up as a constant roll/pitch/yaw torque the rate loop does not expect.

Each model is linear in its unknowns theta, y = Phi theta + noise, so a Kalman filter
with a random-walk model of theta tracks them. The filter's covariance P is its own
measure of uncertainty: small P means "I know these numbers well".

The problem with any fixed filter is the speed/noise trade-off. A filter that trusts its
estimate (small drift) is smooth but takes seconds to notice a sudden change; a filter
that forgets fast is quick but noisy. The change detector resolves this:

    nis = e^T S^-1 e                  normalised innovation squared (chi-square with m dof
                                      when the model is right)
    g   = max(0, g + nis - m - k)     CUSUM statistic: accumulates evidence of a misfit
    if g > h: P <- P + P0, g <- 0     ALARM: "uncertainty has arrived", reopen the estimate

After an alarm the filter is as open-minded as at take-off and re-converges in a few
tenths of a second, then settles back to smooth tracking.
"""
import numpy as np


class KalmanCalibrator:
    """Batched Kalman estimator of theta in y = Phi theta + v, with a CUSUM trigger."""

    def __init__(self, n, theta0, prior_std, drift_std, meas_std,
                 cusum_slack=3.0, cusum_threshold=30.0, trigger=True):
        self.n = n
        self.theta0 = np.asarray(theta0, dtype=float)
        self.p = self.theta0.size
        self.prior_std = np.broadcast_to(np.asarray(prior_std, dtype=float), (self.p,)).copy()
        self.P0 = np.diag(self.prior_std ** 2)
        self.Q = np.diag(np.broadcast_to(np.asarray(drift_std, dtype=float), (self.p,)) ** 2)
        self.meas_std = np.asarray(meas_std, dtype=float)
        self.slack = float(cusum_slack)
        self.threshold = float(cusum_threshold)
        self.trigger = bool(trigger)
        self.theta = np.tile(self.theta0, (n, 1))
        self.P = np.tile(self.P0, (n, 1, 1))
        self.g = np.zeros(n)
        self.nis = np.zeros(n)
        self.alarm = np.zeros(n, dtype=bool)
        self.alarm_count = np.zeros(n, dtype=int)

    def reset(self, idx=None):
        idx = slice(None) if idx is None else idx
        self.theta[idx] = self.theta0
        self.P[idx] = self.P0
        self.g[idx] = 0.0
        self.nis[idx] = 0.0
        self.alarm[idx] = False
        self.alarm_count[idx] = 0

    @property
    def std(self):
        """(n, p) one-sigma uncertainty of each parameter."""
        return np.sqrt(np.maximum(np.diagonal(self.P, axis1=1, axis2=2), 0.0))

    @property
    def relative_std(self):
        """Uncertainty as a fraction of the take-off (prior) uncertainty, in [0, ~1]."""
        return self.std / self.prior_std

    def update(self, Phi, y, active=None):
        """One Kalman step for every drone. Phi (n, m, p), y (n, m). Returns the alarm mask."""
        n, m, p = Phi.shape
        Rm = np.diag(np.broadcast_to(self.meas_std, (m,)) ** 2)
        P = self.P + self.Q                                   # predict: parameters may drift
        e = y - np.einsum("nmp,np->nm", Phi, self.theta)      # innovation (prediction error)
        PhiT = np.transpose(Phi, (0, 2, 1))
        S = Phi @ P @ PhiT + Rm                               # its predicted covariance
        S_inv = np.linalg.inv(S)
        K = P @ PhiT @ S_inv                                  # Kalman gain
        theta = self.theta + np.einsum("npm,nm->np", K, e)
        I_KPhi = np.eye(p) - K @ Phi
        P = I_KPhi @ P @ np.transpose(I_KPhi, (0, 2, 1)) + K @ Rm @ np.transpose(K, (0, 2, 1))
        P = 0.5 * (P + np.transpose(P, (0, 2, 1)))           # Joseph form, kept symmetric

        nis = np.einsum("nm,nmk,nk->n", e, S_inv, e)
        g = np.maximum(0.0, self.g + nis - m - self.slack)
        alarm = (g > self.threshold) if self.trigger else np.zeros(n, dtype=bool)
        if alarm.any():                                       # uncertainty has arrived: reopen
            P[alarm] += self.P0
            g[alarm] = 0.0

        if active is None:
            active = np.ones(n, dtype=bool)
        self.theta[active] = theta[active]
        self.P[active] = P[active]
        self.g[active] = g[active]
        self.nis[active] = nis[active]
        self.alarm = alarm & active
        self.alarm_count += self.alarm
        return self.alarm


def make_calibrators(n, cfg, nominal_mass, trigger=None):
    """The translational and rotational estimators, built from the `calibrator` config section."""
    c = cfg
    trig = c["trigger"] if trigger is None else trigger
    c0 = 1.0 / nominal_mass
    lin = KalmanCalibrator(
        n,
        theta0=[c0, 0.0, 0.0, 0.0],
        prior_std=[c["thrust_gain_prior_std"], *c["disturbance_prior_std"]],
        drift_std=[c["thrust_gain_drift"], *c["disturbance_drift"]],
        meas_std=c["accel_meas_std"],
        cusum_slack=c["cusum_slack"], cusum_threshold=c["cusum_threshold"], trigger=trig,
    )
    ang = KalmanCalibrator(
        n,
        theta0=[0.0, 0.0, 0.0],
        prior_std=c["torque_prior_std"],
        drift_std=c["torque_drift"],
        meas_std=c["torque_meas_std"],
        cusum_slack=c["cusum_slack"], cusum_threshold=c["cusum_threshold"], trigger=trig,
    )
    return lin, ang
