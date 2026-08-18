"""
Euler angle <-> rotation matrix conversions and skew-symmetric helpers.

Convention (docs/control_system/control_system.tex, section "Attitude
Controller"), matching the rigid body simulator:

    R_nb = R_z(psi) @ R_y(theta) @ R_x(phi)

transforms BODY-frame coordinates into the WORLD frame:

    v_n = R_nb @ v_b

Beware: simulations.ahrs.euler.rotation_matrix_to_euler uses the TRANSPOSED
convention (world -> body). To reuse it here you would have to pass R_nb.T.
"""
from __future__ import annotations

import numpy as np

GIMBAL_LOCK_THRESHOLD = 1.0 - 1e-6


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build R_nb (body -> world) from ZYX Euler angles in radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    R_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return R_z @ R_y @ R_x


def rotation_matrix_to_euler(
    R: np.ndarray,
    gimbal_threshold: float = GIMBAL_LOCK_THRESHOLD,
) -> tuple[float, float, float]:
    """Extract ZYX Euler angles (roll, pitch, yaw) from R_nb (body -> world).

    Inverse of euler_to_rotation_matrix. At gimbal lock (pitch = +/-90 deg)
    roll and yaw are not separable; roll is pinned to zero and the whole
    rotation is attributed to yaw.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R.shape}")

    r31 = np.clip(R[2, 0], -1.0, 1.0)

    if r31 <= -gimbal_threshold:
        # Pitch up: theta = +pi/2
        pitch = np.pi / 2.0
        roll = 0.0
        yaw = np.arctan2(R[1, 2], R[1, 1])
    elif r31 >= gimbal_threshold:
        # Pitch down: theta = -pi/2
        pitch = -np.pi / 2.0
        roll = 0.0
        yaw = np.arctan2(-R[1, 2], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = -np.arcsin(r31)
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return float(roll), float(pitch), float(yaw)


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]_x satisfying [v]_x @ w == cross(v, w)."""
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"v must have shape (3,), got {v.shape}")
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def unskew(S: np.ndarray) -> np.ndarray:
    """Inverse of skew(): the vee operator on a skew-symmetric matrix.

    Reads the [S32, S13, S21] components of the spec's vee operator without
    checking antisymmetry, so a non-skew input silently returns its
    lower-triangle entries. Callers antisymmetrise first, as the attitude
    controller does with 0.5 * (R_e - R_e^T).
    """
    S = np.asarray(S, dtype=float)
    if S.shape != (3, 3):
        raise ValueError(f"S must have shape (3, 3), got {S.shape}")
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (or array of angles) into [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
