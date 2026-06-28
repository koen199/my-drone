"""
AHRS filter orchestrator.

Rotation matrix convention used throughout this module:
    R_nb transforms navigation-frame vectors into the body frame:
        v_body = R_nb @ v_nav
    To go the other way:
        v_nav = R_nb.T @ v_body
"""
from __future__ import annotations
import numpy as np

from .triad import triad_init
from .gyro_integrator import integrate_gyro
from .ema_filter import EMAFilter
from .drift_corrector import compute_gravity_correction, compute_mag_correction, apply_drift_correction
from .euler import rotation_matrix_to_euler
from .math_utils import gram_schmidt


class AHRSFilter:
    """Attitude and Heading Reference System filter.

    Fuses gyroscope, accelerometer, and magnetometer data to estimate drone
    orientation as a rotation matrix (R_nb) and ZYX Euler angles.

    Parameters
    ----------
    dt : float
        Sample period in seconds (default 0.01 for 100 Hz).
    gravity_gain : float
        Roll/pitch drift correction gain K_g (default 0.01).
    mag_gain : float
        Yaw drift correction gain K_m (default 0.01).
    ema_window_accel_s : float
        EMA window for the accelerometer reference in seconds (default 60.0).
    ema_window_mag_s : float
        EMA window for the magnetometer reference in seconds (default 60.0).
    ortho_period : int
        Steps between Gram-Schmidt re-orthogonalisations (default 1 = every step).
    """

    def __init__(
        self,
        dt: float = 0.01,
        gravity_gain: float = 0.01,
        mag_gain: float = 0.01,
        ema_window_accel_s: float = 60.0,
        ema_window_mag_s: float = 60.0,
        ortho_period: int = 1,
    ) -> None:
        self._dt = dt
        self._gravity_gain = gravity_gain
        self._mag_gain = mag_gain
        self._ema_window_accel_s = ema_window_accel_s
        self._ema_window_mag_s = ema_window_mag_s
        self._ortho_period = ortho_period

        self._R: np.ndarray | None = None
        self._euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._step: int = 0
        self._accel_ema: EMAFilter | None = None
        self._mag_ema: EMAFilter | None = None
        self._gravity_ref_body: np.ndarray = np.zeros(3)
        self._mag_ref_body: np.ndarray = np.zeros(3)

    def initialize(self, accel_body: np.ndarray, mag_body: np.ndarray) -> None:
        """Establish the initial orientation via TRIAD and seed the EMA filters.

        Must be called once before the first update().

        Parameters
        ----------
        accel_body : ndarray, shape (3,)
            Static accelerometer reading in the body frame (m/s^2).
        mag_body : ndarray, shape (3,)
            Static magnetometer reading in the body frame.
        """
        self._R = triad_init(accel_body, mag_body)
        # EMAFilter stores its estimate in the navigation frame, so convert the
        # initial body-frame readings using the just-computed R_nb.
        accel_nav = self._R.T @ accel_body
        mag_nav = self._R.T @ mag_body
        self._accel_ema = EMAFilter.from_window_seconds(
            self._ema_window_accel_s, self._dt, initial_value=accel_nav
        )
        self._mag_ema = EMAFilter.from_window_seconds(
            self._ema_window_mag_s, self._dt, initial_value=mag_nav
        )
        self._euler = rotation_matrix_to_euler(self._R)
        self._step = 0

    def update(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: np.ndarray,
    ) -> tuple[float, float, float]:
        """Run one AHRS update cycle.

        Parameters
        ----------
        gyro : ndarray, shape (3,)
            Angular velocity [wx, wy, wz] in rad/s (body frame).
        accel : ndarray, shape (3,)
            Accelerometer reading in m/s^2 (body frame).
        mag : ndarray, shape (3,)
            Magnetometer reading in body frame.

        Returns
        -------
        roll, pitch, yaw : float
            Current Euler angles in radians.
        """
        if self._R is None:
            raise RuntimeError("Call initialize() before update().")

        # 1. Gyroscope integration
        R = integrate_gyro(self._R, gyro, self._dt)

        # 2. Update EMA reference vectors
        gravity_ref = self._accel_ema.update(accel, R)
        mag_ref = self._mag_ema.update(mag, R)
        self._gravity_ref_body = gravity_ref
        self._mag_ref_body = mag_ref

        # 3. Drift correction
        R_g = compute_gravity_correction(R, gravity_ref, self._gravity_gain)
        R_m = compute_mag_correction(R, mag_ref, self._mag_gain)
        R = apply_drift_correction(R, R_g, R_m)

        # 4. Periodic Gram-Schmidt re-orthogonalisation
        self._step += 1
        if self._step % self._ortho_period == 0:
            R = gram_schmidt(R)

        self._R = R
        self._euler = rotation_matrix_to_euler(R)
        return self._euler

    @property
    def rotation_matrix(self) -> np.ndarray:
        if self._R is None:
            raise RuntimeError("Call initialize() before accessing rotation_matrix.")
        return self._R.copy()

    @property
    def euler_angles(self) -> tuple[float, float, float]:
        return self._euler

    @property
    def gravity_ref_body(self) -> np.ndarray:
        return self._gravity_ref_body.copy()

    @property
    def mag_ref_body(self) -> np.ndarray:
        return self._mag_ref_body.copy()

    @property
    def step_count(self) -> int:
        return self._step
