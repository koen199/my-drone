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
from .drift_corrector import compute_orientation_error, compute_correction, apply_drift_correction
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
    correction_gain : float
        Drift correction gain beta, with 0 < beta << 1 (default 0.01). A fraction
        beta of the TRIAD-vs-gyro orientation error is applied each update.
    ortho_period : int
        Steps between Gram-Schmidt re-orthogonalisations (default 1 = every step).
    """

    def __init__(
        self,
        dt: float = 0.01,
        correction_gain: float = 0.01,
        ortho_period: int = 1,
    ) -> None:
        self._dt = dt
        self._correction_gain = correction_gain
        self._ortho_period = ortho_period

        self._R: np.ndarray | None = None
        self._euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._step: int = 0
        self._R_triad: np.ndarray | None = None

    def initialize(self, accel_body: np.ndarray, mag_body: np.ndarray) -> None:
        """Establish the initial orientation via TRIAD.

        Must be called once before the first update().

        Parameters
        ----------
        accel_body : ndarray, shape (3,)
            Static accelerometer reading in the body frame (m/s^2).
        mag_body : ndarray, shape (3,)
            Static magnetometer reading in the body frame.
        """
        self._R = triad_init(accel_body, mag_body)
        self._R_triad = self._R.copy()
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

        # 1. Gyroscope integration propagates the current orientation estimate.
        R_gyro = integrate_gyro(self._R, gyro, self._dt)
        R = R_gyro

        # 2. Absolute orientation from raw, normalized accel/mag via TRIAD.
        #    triad_init normalizes its inputs internally. It can fail when the
        #    measurements are degenerate (near-zero or parallel); in that case the
        #    correction is skipped for this step and only the gyro estimate is kept.
        try:
            R_triad = triad_init(accel, mag)
        except ValueError:
            R_triad = None

        # 3. Partial drift correction toward the TRIAD estimate.
        if R_triad is not None:
            self._R_triad = R_triad
            R_err = compute_orientation_error(R_triad, R_gyro)
            R_corr = compute_correction(R_err, self._correction_gain)
            R = apply_drift_correction(R_corr, R_gyro)

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
    def triad_matrix(self) -> np.ndarray | None:
        """Most recent absolute orientation estimate from TRIAD, or None."""
        return None if self._R_triad is None else self._R_triad.copy()

    @property
    def step_count(self) -> int:
        return self._step
