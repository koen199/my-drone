import numpy as np
from .math_utils import skew_symmetric


def integrate_gyro(R: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Propagate rotation matrix one timestep via first-order Taylor expansion.

    R_{k+1} = (I + Omega(omega) * dt) @ R_k

    Parameters
    ----------
    R : ndarray, shape (3,3)
        Current rotation matrix R_nb (nav-to-body).
    omega : ndarray, shape (3,)
        Angular velocity in the body frame (rad/s).
    dt : float
        Timestep in seconds.

    Returns
    -------
    R_new : ndarray, shape (3,3)
        Propagated rotation matrix (not yet re-orthogonalised).
    """
    return (np.eye(3) + skew_symmetric(omega) * dt) @ R
