import numpy as np

GIMBAL_LOCK_THRESHOLD = 1.0 - 1e-6


def rotation_matrix_to_euler(
    R: np.ndarray,
    gimbal_threshold: float = GIMBAL_LOCK_THRESHOLD,
) -> tuple[float, float, float]:
    """Extract ZYX Euler angles (roll, pitch, yaw) from rotation matrix R_nb.

    Convention: R_nb transforms nav-frame vectors into the body frame.
    ZYX sequence: yaw first, then pitch, then roll.

    Parameters
    ----------
    R : ndarray, shape (3,3)
    gimbal_threshold : float
        |r13| threshold above which gimbal lock handling activates.

    Returns
    -------
    roll, pitch, yaw : float
        Euler angles in radians.
    """
    r13 = np.clip(R[0, 2], -1.0, 1.0)

    if r13 >= gimbal_threshold:
        # Pitch down: theta = -pi/2
        pitch = -np.pi / 2.0
        roll = 0.0
        yaw = -np.arctan2(R[1, 0], R[1, 1])
    elif r13 <= -gimbal_threshold:
        # Pitch up: theta = +pi/2
        pitch = np.pi / 2.0
        roll = 0.0
        yaw = np.arctan2(R[1, 0], R[1, 1])
    else:
        roll = np.arctan2(R[1, 2], R[2, 2])
        pitch = -np.arcsin(r13)
        yaw = np.arctan2(R[0, 1], R[0, 0])

    return float(roll), float(pitch), float(yaw)
