import numpy as np
from .math_utils import rotation_log, rotation_exp


def compute_orientation_error(R_triad: np.ndarray, R_gyro: np.ndarray) -> np.ndarray:
    """Rotational difference between the TRIAD and gyroscope orientation estimates.

    R_err = R_triad @ R_gyro.T is the rotation required to align the gyroscope
    estimate with the absolute (TRIAD) orientation estimate. Equals the identity
    when the two estimates coincide.

    Parameters
    ----------
    R_triad : ndarray, shape (3,3)
        Absolute orientation estimate from the TRIAD algorithm (R_nb).
    R_gyro : ndarray, shape (3,3)
        Propagated gyroscope orientation estimate (R_nb).

    Returns
    -------
    R_err : ndarray, shape (3,3)
        Orientation error rotation matrix.
    """
    return R_triad @ R_gyro.T


def compute_correction(R_err: np.ndarray, gain: float) -> np.ndarray:
    """Scale the orientation error down to a partial correction rotation.

    The full error R_err is mapped to its rotation vector phi via the matrix
    logarithm, scaled by the correction gain, and mapped back to a rotation:
        R_corr = exp(beta * log(R_err)).

    Parameters
    ----------
    R_err : ndarray, shape (3,3)
        Orientation error rotation matrix from compute_orientation_error().
    gain : float
        Correction gain beta, with 0 < beta << 1. For example beta = 0.01 applies
        roughly one percent of the estimated error per update.

    Returns
    -------
    R_corr : ndarray, shape (3,3)
        Partial correction rotation matrix.
    """
    phi = rotation_log(R_err)
    return rotation_exp(gain * phi)


def apply_drift_correction(R_corr: np.ndarray, R_gyro: np.ndarray) -> np.ndarray:
    """Apply the partial correction to the gyroscope orientation estimate.

    Returns R_corr @ R_gyro (pre-multiplication in the navigation frame).
    """
    return R_corr @ R_gyro
