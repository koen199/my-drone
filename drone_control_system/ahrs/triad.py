import numpy as np
from .math_utils import safe_normalize


def triad_init(accel_body: np.ndarray, mag_body: np.ndarray) -> np.ndarray:
    """Compute the initial rotation matrix R_nb from static sensor readings.

    Convention: R_nb transforms navigation-frame vectors into the body frame,
    i.e. v_body = R_nb @ v_nav.

    Parameters
    ----------
    accel_body : ndarray, shape (3,)
        Accelerometer reading in the body frame (m/s^2). Points upward (+Z) when static.
    mag_body : ndarray, shape (3,)
        Magnetometer reading in the body frame.

    Returns
    -------
    R_nb : ndarray, shape (3,3)
        Initial rotation matrix.
    """
    u_b = safe_normalize(accel_body)               # Up (tracks nav +Z)
    e_b = safe_normalize(np.cross(mag_body, u_b))  # East
    n_b = np.cross(u_b, e_b)                       # North

    # Columns are the nav ENU axes expressed in body frame → this matrix IS R_nb (nav-to-body).
    return np.column_stack([e_b, n_b, u_b])
