"""
Attitude controller (docs/control_system/control_system.tex, section
"Attitude Controller").

The measured and desired attitudes are turned into rotation matrices R_m and
R_d (body -> world, ZYX Euler sequence). The relative rotation error,
expressed in the body frame, is

    R_e = R_m^T R_d

and satisfies R_m R_e = R_d, i.e. R_e is the rotation that still has to be
applied to the measured body frame to reach the desired one.

Rather than extracting the axis-angle pair (u, theta) — which needs an arccos
and a division by 2 sin(theta) that is singular exactly at the operating point
theta = 0 — we take the antisymmetric part of R_e directly. By Rodrigues'
formula R_e - R_e^T = 2 sin(theta) [u]_x, so

    e_R = 0.5 ( R_e - R_e^T )^v = sin(theta) u

which equals theta*u to first order for small errors, needs no divisions or
trigonometry, and is well defined at theta = 0. The angular velocity setpoint
is then simply

    omega_sp = Kp e_R                      (body frame, rad/s)
"""
from __future__ import annotations

import numpy as np

from .euler import euler_to_rotation_matrix, unskew


def orientation_error(R_measured: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
    """Geometric orientation error e_R = sin(theta) * u, in the body frame.

    Parameters
    ----------
    R_measured, R_desired : ndarray, shape (3, 3)
        Body -> world rotation matrices R_m and R_d.

    Notes
    -----
    e_R vanishes both at zero error (theta = 0) and at theta = 180 deg, where
    R_e is symmetric. The 180 deg case is an unstable equilibrium: any
    perturbation off it produces a non-zero error that drives the vehicle the
    rest of the way. The magnitude also folds back for theta > 90 deg, so
    large attitude steps rotate at less than the full proportional rate.
    """
    R_measured = np.asarray(R_measured, dtype=float)
    R_desired = np.asarray(R_desired, dtype=float)
    if R_measured.shape != (3, 3) or R_desired.shape != (3, 3):
        raise ValueError("R_measured and R_desired must both have shape (3, 3)")

    R_error = R_measured.T @ R_desired
    return 0.5 * unskew(R_error - R_error.T)


class AttitudeController:
    """P controller from attitude error to a body-frame angular rate setpoint.

    Parameters
    ----------
    kp : float or array_like, shape (3,)
        Proportional gain in 1/s. A 3-vector applies per-axis gains in the
        body frame (roll, pitch, yaw), which is the usual way to make yaw
        slower than roll and pitch.
    max_rate : float or array_like, optional
        Symmetric saturation on the commanded body rate, in rad/s. Keeps large
        attitude steps from demanding rates the rate loop cannot deliver.
    """

    def __init__(self, kp, max_rate=None) -> None:
        kp = np.asarray(kp, dtype=float)
        self._kp = np.full(3, float(kp)) if kp.ndim == 0 else kp.copy()
        if self._kp.shape != (3,):
            raise ValueError(f"kp must be a scalar or have shape (3,), got {self._kp.shape}")
        if np.any(self._kp < 0.0):
            raise ValueError("kp must be >= 0")

        if max_rate is None:
            self._max_rate = None
        else:
            max_rate = np.asarray(max_rate, dtype=float)
            self._max_rate = np.abs(np.full(3, float(max_rate)) if max_rate.ndim == 0 else max_rate.copy())
            if self._max_rate.shape != (3,):
                raise ValueError("max_rate must be a scalar or have shape (3,)")

    def update(self, R_measured: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
        """Return the body-frame angular velocity setpoint omega_sp in rad/s."""
        rate_setpoint = self._kp * orientation_error(R_measured, R_desired)
        if self._max_rate is not None:
            rate_setpoint = np.clip(rate_setpoint, -self._max_rate, self._max_rate)
        return rate_setpoint

    def update_from_euler(
        self,
        measured: tuple[float, float, float],
        desired: tuple[float, float, float],
    ) -> np.ndarray:
        """Same as update(), taking (roll, pitch, yaw) pairs in radians."""
        return self.update(
            euler_to_rotation_matrix(*measured),
            euler_to_rotation_matrix(*desired),
        )

    @property
    def kp(self) -> np.ndarray:
        return self._kp.copy()
