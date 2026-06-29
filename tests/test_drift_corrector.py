import numpy as np
import pytest
from drone_control_system.ahrs.drift_corrector import (
    compute_orientation_error,
    compute_correction,
    apply_drift_correction,
)
from drone_control_system.ahrs.math_utils import rodrigues_rotation, rotation_log


def _is_rotation(R):
    return np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_orientation_error_identity_when_estimates_agree():
    R = rodrigues_rotation(np.array([1.0, 2.0, 3.0]), 0.7)
    R_err = compute_orientation_error(R, R)
    assert np.allclose(R_err, np.eye(3), atol=1e-12)


def test_orientation_error_recovers_relative_rotation():
    R_gyro = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), 0.3)
    delta = rodrigues_rotation(np.array([1.0, 0.0, 0.0]), 0.1)
    R_triad = delta @ R_gyro
    R_err = compute_orientation_error(R_triad, R_gyro)
    assert np.allclose(R_err, delta, atol=1e-12)


def test_correction_identity_when_no_error():
    R_corr = compute_correction(np.eye(3), gain=0.01)
    assert np.allclose(R_corr, np.eye(3), atol=1e-12)


def test_correction_is_rotation_matrix():
    R_err = rodrigues_rotation(np.array([0.2, -0.5, 0.3]), 0.4)
    R_corr = compute_correction(R_err, gain=0.01)
    assert _is_rotation(R_corr)


def test_correction_applies_gain_fraction_of_error():
    axis = np.array([0.0, 1.0, 0.0])
    angle = 0.2
    R_err = rodrigues_rotation(axis, angle)
    beta = 0.01
    R_corr = compute_correction(R_err, gain=beta)
    # The correction rotation vector should be beta times the error rotation vector.
    phi_corr = rotation_log(R_corr)
    assert np.allclose(phi_corr, beta * rotation_log(R_err), atol=1e-9)
    assert np.isclose(np.linalg.norm(phi_corr), beta * angle, atol=1e-9)


def test_correction_gain_scaling():
    R_err = rodrigues_rotation(np.array([1.0, 0.0, 0.0]), 0.3)
    angle = lambda Rc: np.linalg.norm(rotation_log(Rc))
    assert angle(compute_correction(R_err, gain=0.5)) > angle(compute_correction(R_err, gain=0.01))


def test_apply_drift_correction_identity_case():
    R_gyro = rodrigues_rotation(np.array([1.0, 1.0, 0.0]), 0.5)
    assert np.allclose(apply_drift_correction(np.eye(3), R_gyro), R_gyro)


def test_apply_drift_correction_composition():
    R_gyro = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), 0.2)
    R_corr = rodrigues_rotation(np.array([1.0, 0.0, 0.0]), 0.01)
    assert np.allclose(apply_drift_correction(R_corr, R_gyro), R_corr @ R_gyro)


def test_partial_correction_reduces_error():
    # A small fraction of the correction moves the gyro estimate toward TRIAD.
    R_gyro = np.eye(3)
    R_triad = rodrigues_rotation(np.array([0.3, -0.2, 0.5]), 0.15)
    R_err = compute_orientation_error(R_triad, R_gyro)
    R_corr = compute_correction(R_err, gain=0.1)
    R_new = apply_drift_correction(R_corr, R_gyro)

    err_before = np.linalg.norm(rotation_log(compute_orientation_error(R_triad, R_gyro)))
    err_after = np.linalg.norm(rotation_log(compute_orientation_error(R_triad, R_new)))
    assert err_after < err_before
