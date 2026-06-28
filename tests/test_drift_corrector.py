import numpy as np
import pytest
from drone_control_system.ahrs.drift_corrector import (
    compute_gravity_correction,
    compute_mag_correction,
    apply_drift_correction,
    GRAVITY_NAV,
    MAG_NORTH_NAV,
)


def _angle_between(a, b):
    cos_theta = np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1)
    return np.arccos(cos_theta)


def test_gravity_correction_no_error():
    # When reference matches theoretical, correction should be identity
    R = np.eye(3)
    g_ref = R @ GRAVITY_NAV  # same as theoretical in body frame
    R_g = compute_gravity_correction(R, g_ref, gain=0.01)
    assert np.allclose(R_g, np.eye(3), atol=1e-10)


def test_gravity_correction_is_rotation_matrix():
    R = np.eye(3)
    g_ref = np.array([0.1, 0.0, -9.81])  # slightly off
    R_g = compute_gravity_correction(R, g_ref, gain=0.01)
    assert np.allclose(R_g @ R_g.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R_g), 1.0, atol=1e-6)


def test_gravity_correction_reduces_error():
    R = np.eye(3)
    g_theoretical = R @ GRAVITY_NAV
    # Reference slightly off from theoretical (small lateral component)
    g_ref = np.array([0.5, 0.0, 9.81])
    g_ref /= np.linalg.norm(g_ref)
    g_ref *= np.linalg.norm(GRAVITY_NAV)

    angle_before = _angle_between(g_theoretical, g_ref)
    R_g = compute_gravity_correction(R, g_ref, gain=0.5)
    g_corrected = R_g @ g_theoretical
    angle_after = _angle_between(g_corrected, g_ref)

    assert angle_after < angle_before


def test_gravity_correction_gain_scaling():
    R = np.eye(3)
    g_ref = np.array([0.3, 0.0, 9.81])
    R_low = compute_gravity_correction(R, g_ref, gain=0.01)
    R_high = compute_gravity_correction(R, g_ref, gain=0.5)

    # Small-angle Rodrigues: R = I + θ*skew(axis), so ‖R - I‖_F encodes the angle.
    magnitude = lambda Rc: np.linalg.norm(Rc - np.eye(3))
    assert magnitude(R_high) > magnitude(R_low)


def test_mag_correction_no_error():
    R = np.eye(3)
    mag_ref = R @ MAG_NORTH_NAV
    R_m = compute_mag_correction(R, mag_ref, gain=0.01)
    assert np.allclose(R_m, np.eye(3), atol=1e-10)


def test_apply_drift_correction_identity_case():
    R = np.eye(3)
    result = apply_drift_correction(R, np.eye(3), np.eye(3))
    assert np.allclose(result, np.eye(3))


def test_apply_drift_correction_composition():
    from drone_control_system.ahrs.math_utils import rodrigues_rotation
    R = np.eye(3)
    R_g = rodrigues_rotation(np.array([1.0, 0.0, 0.0]), 0.01)
    R_m = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), 0.005)
    result = apply_drift_correction(R, R_g, R_m)
    assert np.allclose(result, R_g @ R_m @ R)
