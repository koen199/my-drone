import numpy as np


def safe_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector with norm {norm}")
    return v / norm


def skew_symmetric(omega: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix Omega such that Omega @ b == cross(omega, b)."""
    wx, wy, wz = omega
    return np.array([
        [ 0.0, -wz,  wy],
        [ wz,  0.0, -wx],
        [-wy,  wx,  0.0],
    ])


def gram_schmidt(R: np.ndarray) -> np.ndarray:
    """Re-orthogonalise a 3x3 rotation matrix using Gram-Schmidt on its rows."""
    r1 = safe_normalize(R[0])
    r2 = R[1] - np.dot(R[1], r1) * r1
    r2 = safe_normalize(r2)
    r3 = np.cross(r1, r2)
    return np.array([r1, r2, r3])


def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return 3x3 rotation matrix via Rodrigues formula for a unit axis and angle (rad)."""
    k = safe_normalize(axis)
    K = skew_symmetric(k)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rodrigues_small_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Linearised Rodrigues correction: I + angle * skew(axis). Valid for small angles."""
    return np.eye(3) + angle * skew_symmetric(axis)
