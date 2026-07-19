import numpy as np
import pytest
from simulations.ahrs.triad import triad_init


def test_triad_output_is_rotation_matrix():
    accel = np.array([0.1, 0.0, 9.81])
    mag = np.array([0.2, 0.9, 0.1])
    R = triad_init(accel, mag)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_triad_level_drone_near_identity():
    # Level drone: accel points straight up (+Z), mag points north (+Y in ENU)
    accel = np.array([0.0, 0.0, 9.81])
    mag = np.array([0.0, 1.0, 0.0])
    R = triad_init(accel, mag)
    assert np.allclose(R, np.eye(3), atol=1e-10)


def test_triad_orthogonality_of_basis():
    accel = np.array([0.5, 0.2, 9.5])
    mag = np.array([0.1, 0.8, 0.3])
    R = triad_init(accel, mag)
    # Rows of R_nb correspond to columns of R_gb (e, n, u) — must be orthonormal
    for i in range(3):
        assert np.isclose(np.linalg.norm(R[i]), 1.0, atol=1e-10)
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert np.isclose(np.dot(R[i], R[j]), expected, atol=1e-10)


def test_triad_up_column():
    # Third column of R_nb must be u_b: nav +Z maps to the accel direction in body.
    # Test both level and tilted to catch any transpose mistake.
    for accel, mag in [
        (np.array([0.0, 0.0, 9.81]), np.array([0.3, 0.9, 0.1])),
        (np.array([0.0, 4.905, 8.503]), np.array([0.0, 0.5, 0.5])),  # ~30° roll
    ]:
        R = triad_init(accel, mag)
        u_b = accel / np.linalg.norm(accel)
        assert np.allclose(R[:, 2], u_b, atol=1e-10)


def test_triad_raises_on_degenerate_input():
    # accel parallel to mag causes zero cross product
    v = np.array([0.0, 0.0, 1.0])
    with pytest.raises((ValueError, np.linalg.LinAlgError, ZeroDivisionError, Exception)):
        triad_init(v, v)
