import numpy as np
import pytest
from simulations.ahrs.euler import rotation_matrix_to_euler, GIMBAL_LOCK_THRESHOLD


def _Rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])

def _Ry(a):
    return np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])

def _Rz(a):
    return np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_euler_identity_matrix():
    roll, pitch, yaw = rotation_matrix_to_euler(np.eye(3))
    assert np.isclose(roll, 0.0) and np.isclose(pitch, 0.0) and np.isclose(yaw, 0.0)


def test_euler_pure_roll_90():
    R = _Rx(np.pi / 2)
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    assert np.isclose(roll, np.pi / 2, atol=1e-10)
    assert np.isclose(pitch, 0.0, atol=1e-10)
    assert np.isclose(yaw, 0.0, atol=1e-10)


def test_euler_pure_pitch_45():
    R = _Ry(np.pi / 4)
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    assert np.isclose(roll, 0.0, atol=1e-10)
    assert np.isclose(pitch, np.pi / 4, atol=1e-10)
    assert np.isclose(yaw, 0.0, atol=1e-10)


def test_euler_pure_yaw_90():
    R = _Rz(np.pi / 2)
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    assert np.isclose(roll, 0.0, atol=1e-10)
    assert np.isclose(pitch, 0.0, atol=1e-10)
    assert np.isclose(yaw, np.pi / 2, atol=1e-10)


def test_euler_round_trip():
    phi, theta, psi = 0.3, 0.2, 1.1
    R = _Rx(phi) @ _Ry(theta) @ _Rz(psi)
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    assert np.isclose(roll, phi, atol=1e-10)
    assert np.isclose(pitch, theta, atol=1e-10)
    assert np.isclose(yaw, psi, atol=1e-10)


def test_euler_gimbal_lock_pitch_up():
    # r13 = -1 => pitch = +pi/2
    R = _Ry(np.pi / 2)
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    assert np.isclose(pitch, np.pi / 2, atol=1e-6)
    assert np.isclose(roll, 0.0, atol=1e-6)


def test_euler_gimbal_lock_pitch_down():
    # r13 = +1 => pitch = -pi/2
    R = _Ry(-np.pi / 2)
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    assert np.isclose(pitch, -np.pi / 2, atol=1e-6)
    assert np.isclose(roll, 0.0, atol=1e-6)


def test_euler_near_gimbal_lock_no_nan():
    # Just below threshold — should use normal formula without NaN
    pitch_angle = np.arcsin(GIMBAL_LOCK_THRESHOLD - 0.01)
    R = _Ry(pitch_angle)
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    assert not np.isnan(roll) and not np.isnan(pitch) and not np.isnan(yaw)
    assert not np.isinf(roll) and not np.isinf(pitch) and not np.isinf(yaw)


def test_euler_returns_floats():
    roll, pitch, yaw = rotation_matrix_to_euler(np.eye(3))
    assert isinstance(roll, float)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)
