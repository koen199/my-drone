"""
Error-State Extended Kalman Filter of docs/state_estimator/state_estimator.tex.

Nominal state (16 elements):
    x = [p (3), v (3), q (4), b_g (3), b_a (3)]
with p, v in the world/navigation frame n (ENU, z up), q the scalar-first
unit quaternion n<-b, and b_g, b_a the gyroscope and accelerometer biases.

Error state (15 elements), in this index order throughout:
    dx = [dp 0:3, dv 3:6, dtheta 6:9, db_g 9:12, db_a 12:15]
The doc's Q-matrix section lists b_a before b_g; the order above follows the
error-state definition and the F derivation, and is used consistently here.

The filter knows nothing about the simulator: it consumes raw IMU samples in
predict() and raw sensor readings in the update_*() methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulations.math_helper import (
    quaternion_exp,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
    skew,
)

# Error-state slices.
P_IDX = slice(0, 3)
V_IDX = slice(3, 6)
THETA_IDX = slice(6, 9)
BG_IDX = slice(9, 12)
BA_IDX = slice(12, 15)
ERROR_STATE_SIZE = 15


@dataclass
class EskfNoise:
    """Filter tuning: process noise PSDs and measurement noise std devs.

    Process noise (continuous PSDs, the diagonal blocks of Q_c):
        sigma_a  accelerometer white noise       [m/s^2/sqrt(Hz)]
        sigma_g  gyroscope white noise           [rad/s/sqrt(Hz)]
        sigma_ba accelerometer bias random walk  [m/s^3/sqrt(Hz)]
        sigma_bg gyroscope bias random walk      [rad/s^2/sqrt(Hz)]

    Measurement noise (discrete standard deviations, R = sigma^2 I):
        sigma_gps_pos [m], sigma_gps_vel [m/s], sigma_baro [m],
        sigma_mag [field units of mag_reference], sigma_acc [m/s^2]

    acc_gate: the accelerometer gravity update is skipped when
    | ||z_acc|| - g | exceeds this, i.e. when the vehicle is clearly not
    quasi-static.
    """

    sigma_a: float = 2.0e-3
    sigma_g: float = 3.0e-4
    sigma_ba: float = 1.0e-3
    sigma_bg: float = 1.0e-4
    sigma_gps_pos: float = 0.5
    sigma_gps_vel: float = 0.05
    sigma_baro: float = 0.1
    sigma_mag: float = 0.005
    sigma_acc: float = 0.5
    acc_gate: float = 0.3
    gravity: float = 9.81
    mag_reference: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.22, -0.42])
    )


@dataclass
class NominalState:
    """The 16-element nominal state, as separate named vectors."""

    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    b_g: np.ndarray
    b_a: np.ndarray

    def copy(self) -> NominalState:
        return NominalState(
            self.p.copy(), self.v.copy(), self.q.copy(), self.b_g.copy(), self.b_a.copy()
        )


class ErrorStateKalmanFilter:
    """ES-EKF for position, velocity, attitude and IMU biases.

    Typical use::

        eskf = ErrorStateKalmanFilter(EskfNoise())
        eskf.initialize(p0, v0, accel_body, mag_body)
        loop:
            eskf.predict(gyro, accel, dt)          # every IMU sample
            eskf.update_gps(pos, vel)              # whenever a sensor reports
            eskf.update_baro(alt)
            eskf.update_mag(mag_body)
            eskf.update_accel(accel_body)
    """

    def __init__(self, noise: EskfNoise | None = None) -> None:
        self.noise = noise or EskfNoise()
        self._g_n = np.array([0.0, 0.0, -self.noise.gravity])
        self._m_n = np.asarray(self.noise.mag_reference, dtype=float)
        self._x = NominalState(
            p=np.zeros(3),
            v=np.zeros(3),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            b_g=np.zeros(3),
            b_a=np.zeros(3),
        )
        self._P = np.eye(ERROR_STATE_SIZE)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        accel_body: np.ndarray,
        mag_body: np.ndarray,
        sigma_p0: float = 0.5,
        sigma_v0: float = 0.1,
        sigma_theta0: float = np.deg2rad(2.0),
        sigma_bg0: float = 0.02,
        sigma_ba0: float = 0.2,
    ) -> None:
        """Set the nominal state and the initial covariance P0.

        The attitude is solved from a quasi-static accelerometer reading
        (which points body-up, -R^T g_n) and a magnetometer reading
        (R^T m_n) with the TRIAD construction; biases start at zero.
        """
        R_nb = self._triad(np.asarray(accel_body, float), np.asarray(mag_body, float))
        self._x = NominalState(
            p=np.asarray(position, dtype=float).copy(),
            v=np.asarray(velocity, dtype=float).copy(),
            q=rotation_matrix_to_quaternion(R_nb),
            b_g=np.zeros(3),
            b_a=np.zeros(3),
        )
        self._P = np.diag(np.concatenate([
            np.full(3, sigma_p0**2),
            np.full(3, sigma_v0**2),
            np.full(3, sigma_theta0**2),
            np.full(3, sigma_bg0**2),
            np.full(3, sigma_ba0**2),
        ]))

    def _triad(self, accel_body: np.ndarray, mag_body: np.ndarray) -> np.ndarray:
        """R_nb from matching the up and magnetic directions in both frames."""

        def frame(up: np.ndarray, mag: np.ndarray) -> np.ndarray:
            t1 = up / np.linalg.norm(up)
            t2 = np.cross(t1, mag)
            t2 /= np.linalg.norm(t2)
            return np.column_stack([t1, t2, np.cross(t1, t2)])

        T_n = frame(-self._g_n, self._m_n)
        T_b = frame(accel_body, mag_body)
        return T_n @ T_b.T

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """Propagate nominal state and error covariance over one IMU sample.

        gyro is the raw angular rate (rad/s) and accel the raw specific force
        (m/s^2), both body frame, held constant over dt.
        """
        # Error transition Jacobian, evaluated at x_k (before propagation).
        F = self.error_transition(gyro, accel, dt)

        x = self._x
        omega = np.asarray(gyro, dtype=float) - x.b_g
        acc = np.asarray(accel, dtype=float) - x.b_a
        R = quaternion_to_rotation_matrix(x.q)
        I3 = np.eye(3)

        # Nominal state propagation f(x, u).
        x.p = x.p + x.v * dt
        x.v = x.v + (R @ acc + self._g_n) * dt
        x.q = quaternion_normalize(quaternion_multiply(x.q, quaternion_exp(omega * dt)))

        # Q_k = L_c Q_c L_c^T dt; R R^T = I makes it block-diagonal.
        n = self.noise
        Q = np.zeros((ERROR_STATE_SIZE, ERROR_STATE_SIZE))
        Q[V_IDX, V_IDX] = n.sigma_a**2 * dt * I3
        Q[THETA_IDX, THETA_IDX] = n.sigma_g**2 * dt * I3
        Q[BG_IDX, BG_IDX] = n.sigma_bg**2 * dt * I3
        Q[BA_IDX, BA_IDX] = n.sigma_ba**2 * dt * I3

        P = F @ self._P @ F.T + Q
        self._P = 0.5 * (P + P.T)

    def error_transition(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
        """The 15x15 error-state Jacobian F_k at the current nominal state.

        Row blocks from the doc:
            dp:     [I, I dt, 0, 0, 0]
            dv:     [0, I, -R [a]_x dt, 0, -R dt]
            dtheta: [0, 0, I - [w]_x dt, -I dt, 0]
            db_g, db_a: identity
        with a, w the bias-corrected IMU sample.
        """
        x = self._x
        omega = np.asarray(gyro, dtype=float) - x.b_g
        acc = np.asarray(accel, dtype=float) - x.b_a
        R = quaternion_to_rotation_matrix(x.q)
        I3 = np.eye(3)
        F = np.eye(ERROR_STATE_SIZE)
        F[P_IDX, V_IDX] = I3 * dt
        F[V_IDX, THETA_IDX] = -R @ skew(acc) * dt
        F[V_IDX, BA_IDX] = -R * dt
        F[THETA_IDX, THETA_IDX] = I3 - skew(omega) * dt
        F[THETA_IDX, BG_IDX] = -I3 * dt
        return F

    # ------------------------------------------------------------------
    # Measurement models: predicted measurement h(x) and Jacobian H
    # ------------------------------------------------------------------

    def mag_model(self) -> tuple[np.ndarray, np.ndarray]:
        """h = R(q)^T m_n, H_theta = [R(q)^T m_n]_x."""
        z_hat = quaternion_to_rotation_matrix(self._x.q).T @ self._m_n
        H = np.zeros((3, ERROR_STATE_SIZE))
        H[:, THETA_IDX] = skew(z_hat)
        return z_hat, H

    def accel_model(self) -> tuple[np.ndarray, np.ndarray]:
        """h = -R(q)^T g_n + b_a, H_theta = [-R(q)^T g_n]_x, H_ba = I.

        The doc's model omits b_a; it is included here so the measurement
        model matches the bias-corrected specific force used in predict().
        """
        gravity_body = -quaternion_to_rotation_matrix(self._x.q).T @ self._g_n
        H = np.zeros((3, ERROR_STATE_SIZE))
        H[:, THETA_IDX] = skew(gravity_body)
        H[:, BA_IDX] = np.eye(3)
        return gravity_body + self._x.b_a, H

    # ------------------------------------------------------------------
    # Measurement updates
    # ------------------------------------------------------------------

    def _update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray) -> float:
        """Generic ESKF correction; returns the normalised innovation squared.

        Kalman gain, error-state estimate, injection into the nominal state
        and a Joseph-form covariance update, followed by the error reset.
        """
        P = self._P
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T      # P H^T S^-1 (S, P symmetric)
        dx = K @ y

        # State injection x <- x (+) dx.
        x = self._x
        x.p = x.p + dx[P_IDX]
        x.v = x.v + dx[V_IDX]
        x.q = quaternion_normalize(quaternion_multiply(x.q, quaternion_exp(dx[THETA_IDX])))
        x.b_g = x.b_g + dx[BG_IDX]
        x.b_a = x.b_a + dx[BA_IDX]

        I_KH = np.eye(ERROR_STATE_SIZE) - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T
        self._P = 0.5 * (P + P.T)
        # Error reset: dx <- 0 is implicit, the filter never stores dx.
        return float(y @ np.linalg.solve(S, y))

    def update_gps(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """GNSS position and velocity fix in the world frame."""
        H = np.zeros((6, ERROR_STATE_SIZE))
        H[0:3, P_IDX] = np.eye(3)
        H[3:6, V_IDX] = np.eye(3)
        z = np.concatenate([position, velocity])
        y = z - np.concatenate([self._x.p, self._x.v])
        R = np.diag(np.concatenate([
            np.full(3, self.noise.sigma_gps_pos**2),
            np.full(3, self.noise.sigma_gps_vel**2),
        ]))
        return self._update(y, H, R)

    def update_baro(self, altitude: float) -> float:
        """Barometric altitude, world z."""
        H = np.zeros((1, ERROR_STATE_SIZE))
        H[0, 2] = 1.0
        y = np.array([altitude - self._x.p[2]])
        R = np.array([[self.noise.sigma_baro**2]])
        return self._update(y, H, R)

    def update_mag(self, mag_body: np.ndarray) -> float:
        """Magnetometer field vector in the body frame (see mag_model)."""
        z_hat, H = self.mag_model()
        y = np.asarray(mag_body, dtype=float) - z_hat
        R = self.noise.sigma_mag**2 * np.eye(3)
        return self._update(y, H, R)

    def update_accel(self, accel_body: np.ndarray) -> float | None:
        """Accelerometer gravity alignment (quasi-static tilt correction).

        See accel_model. Returns None when the reading fails the
        quasi-static gate | ||z|| - g | > acc_gate.
        """
        z = np.asarray(accel_body, dtype=float)
        if abs(np.linalg.norm(z) - self.noise.gravity) > self.noise.acc_gate:
            return None
        z_hat, H = self.accel_model()
        y = z - z_hat
        R = self.noise.sigma_acc**2 * np.eye(3)
        return self._update(y, H, R)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> NominalState:
        return self._x.copy()

    @state.setter
    def state(self, value: NominalState) -> None:
        """Overwrite the nominal state (covariance is left untouched)."""
        value = value.copy()
        value.q = quaternion_normalize(np.asarray(value.q, dtype=float))
        self._x = value

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    @property
    def sigma(self) -> np.ndarray:
        """Standard deviations of the 15 error states."""
        return np.sqrt(np.diag(self._P))

    @property
    def rotation_matrix(self) -> np.ndarray:
        """R_nb of the nominal attitude."""
        return quaternion_to_rotation_matrix(self._x.q)

    @property
    def euler(self) -> tuple[float, float, float]:
        """ZYX (roll, pitch, yaw) of the nominal attitude."""
        return rotation_matrix_to_euler(self.rotation_matrix)

    def body_rates(self, gyro: np.ndarray) -> np.ndarray:
        """Bias-corrected angular rate for a raw gyro sample."""
        return np.asarray(gyro, dtype=float) - self._x.b_g
