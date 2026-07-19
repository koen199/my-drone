"""
Stateless quaternion math for the rigid body simulator.

Conventions used throughout this package:
    - Quaternions are scalar-first numpy arrays [w, x, y, z] of unit norm.
    - A state quaternion p represents the orientation of the body frame (b)
      relative to the world/navigation frame (n).
    - The rotation matrix derived from p is R_nb, which transforms body-frame
      vectors into the n frame:
          v_n = R_nb @ v_b
      To go the other way:
          v_b = R_nb.T @ v_n
"""
import numpy as np


def quaternion_normalize(q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return q scaled to unit norm. Raises ValueError for a near-zero quaternion."""
    norm = np.linalg.norm(q)
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero quaternion with norm {norm}")
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2 for scalar-first quaternions."""
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    return np.concatenate([
        [w1 * w2 - np.dot(v1, v2)],
        w1 * v2 + w2 * v1 + np.cross(v1, v2),
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate [w, -x, -y, -z]; the inverse for a unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion for a rotation of `angle` (rad) about `axis`."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        raise ValueError(f"Cannot build quaternion from near-zero axis with norm {norm}")
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * (axis / norm)])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R_nb (v_n = R_nb @ v_b) from an orientation quaternion.

    The input is normalized first, so slightly non-unit quaternions (e.g. RK4
    stage states) still produce a proper rotation matrix.
    """
    w, x, y, z = quaternion_normalize(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def quaternion_derivative(q: np.ndarray, omega_n: np.ndarray) -> np.ndarray:
    """Time derivative p_dot = 0.5 * omega_q (x) p for world-frame angular velocity.

    omega_q is the angular velocity promoted to a pure quaternion [0, omega].
    Written out in components (see rigid_body_simulator.tex):
        p_dot = 0.5 * [ -omega . p_vec ;  p0 * omega + omega x p_vec ]
    """
    p0, p_vec = q[0], q[1:]
    return 0.5 * np.concatenate([
        [-np.dot(omega_n, p_vec)],
        p0 * omega_n + np.cross(omega_n, p_vec),
    ])
