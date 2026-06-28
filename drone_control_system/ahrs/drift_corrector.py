import numpy as np
from .math_utils import safe_normalize, skew_symmetric

# Reference vectors in the navigation frame (ENU).
# GRAVITY_NAV is the specific force direction (what the accelerometer reads when
# stationary in ENU), not the gravitational acceleration.  Specific force = -g,
# so it points upward (+Z) in ENU when the drone is level.
GRAVITY_NAV = np.array([0.0, 0.0, 9.81])
MAG_NORTH_NAV = np.array([0.0, 1.0, 0.0])

_EPS = 1e-10


def _small_angle_correction(theoretical_body: np.ndarray,
                             reference_body: np.ndarray,
                             gain: float) -> np.ndarray:
    """Compute a small-angle Rodrigues correction matrix from two body-frame vectors."""
    try:
        t = safe_normalize(theoretical_body)
        r = safe_normalize(reference_body)
    except ValueError:
        return np.eye(3)

    error = np.cross(t, r)
    theta = np.linalg.norm(error)
    if theta < _EPS:
        return np.eye(3)

    axis = error / theta
    corrected_angle = gain * theta
    return np.eye(3) + corrected_angle * skew_symmetric(axis)


def compute_gravity_correction(
    R: np.ndarray,
    gravity_ref_body: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Compute roll/pitch drift correction from accelerometer reference.

    Parameters
    ----------
    R : ndarray, shape (3,3)
        Current rotation matrix R_nb.
    gravity_ref_body : ndarray, shape (3,)
        EMA-filtered gravity estimate in the body frame.
    gain : float
        Correction gain K_g.

    Returns
    -------
    R_g : ndarray, shape (3,3)
        Small-angle correction rotation matrix.
    """
    g_theoretical_body = R @ GRAVITY_NAV
    return _small_angle_correction(g_theoretical_body, gravity_ref_body, gain)


def compute_mag_correction(
    R: np.ndarray,
    mag_ref_body: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Compute yaw drift correction from magnetometer reference.

    Parameters
    ----------
    R : ndarray, shape (3,3)
        Current rotation matrix R_nb.
    mag_ref_body : ndarray, shape (3,)
        EMA-filtered magnetometer estimate in the body frame.
    gain : float
        Correction gain K_m.

    Returns
    -------
    R_m : ndarray, shape (3,3)
        Small-angle yaw correction rotation matrix.
    """
    mag_theoretical_body = R @ MAG_NORTH_NAV
    return _small_angle_correction(mag_theoretical_body, mag_ref_body, gain)


def apply_drift_correction(
    R: np.ndarray,
    R_g: np.ndarray,
    R_m: np.ndarray,
) -> np.ndarray:
    """Apply gravity and magnetic corrections to the propagated orientation.

    Returns R_g @ R_m @ R (pre-multiplication in body frame convention).
    """
    return R_g @ R_m @ R
