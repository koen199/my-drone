import numpy as np
import pytest

from simulations.rigid_body_simulator.quaternion import (
    quaternion_conjugate,
    quaternion_derivative,
    quaternion_from_axis_angle,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    return quaternion_normalize(rng.standard_normal(4))


def _rodrigues_reference(axis: np.ndarray, angle: float) -> np.ndarray:
    """Independent Rodrigues rotation matrix used as a reference in tests."""
    k = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _is_rotation(R: np.ndarray) -> bool:
    return np.allclose(R @ R.T, np.eye(3), atol=1e-10) and np.isclose(
        np.linalg.det(R), 1.0, atol=1e-10
    )


def test_normalize_returns_unit_norm():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    assert np.allclose(quaternion_normalize(q), IDENTITY, atol=1e-12)


def test_normalize_near_zero_raises():
    with pytest.raises(ValueError):
        quaternion_normalize(np.zeros(4))


def test_multiply_identity():
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = _random_unit_quaternion(rng)
        assert np.allclose(quaternion_multiply(IDENTITY, q), q, atol=1e-12)
        assert np.allclose(quaternion_multiply(q, IDENTITY), q, atol=1e-12)


def test_multiply_associative():
    rng = np.random.default_rng(2)
    for _ in range(10):
        q1, q2, q3 = (_random_unit_quaternion(rng) for _ in range(3))
        left = quaternion_multiply(quaternion_multiply(q1, q2), q3)
        right = quaternion_multiply(q1, quaternion_multiply(q2, q3))
        assert np.allclose(left, right, atol=1e-12)


def test_conjugate_is_inverse_for_unit_quaternion():
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = _random_unit_quaternion(rng)
        assert np.allclose(
            quaternion_multiply(q, quaternion_conjugate(q)), IDENTITY, atol=1e-12
        )


def test_from_axis_angle_zero_angle_is_identity():
    q = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.0)
    assert np.allclose(q, IDENTITY, atol=1e-12)


def test_from_axis_angle_zero_axis_raises():
    with pytest.raises(ValueError):
        quaternion_from_axis_angle(np.zeros(3), 0.5)


def test_rotation_matrix_is_orthonormal():
    rng = np.random.default_rng(4)
    for _ in range(10):
        R = quaternion_to_rotation_matrix(_random_unit_quaternion(rng))
        assert _is_rotation(R)


def test_rotation_matrix_matches_rodrigues():
    rng = np.random.default_rng(5)
    for _ in range(10):
        axis = rng.standard_normal(3)
        angle = rng.uniform(-np.pi, np.pi)
        R = quaternion_to_rotation_matrix(quaternion_from_axis_angle(axis, angle))
        assert np.allclose(R, _rodrigues_reference(axis, angle), atol=1e-10)


def test_rotation_matrix_known_rotation():
    # 90 degrees about z maps x_hat onto y_hat.
    q = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    R = quaternion_to_rotation_matrix(q)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-10)


def test_rotation_matrix_normalizes_input():
    q = 3.0 * quaternion_from_axis_angle(np.array([1.0, 1.0, 0.0]), 0.7)
    assert _is_rotation(quaternion_to_rotation_matrix(q))


def test_derivative_orthogonal_to_quaternion():
    # d/dt ||p||^2 = 2 p . p_dot = 0 for a unit quaternion, so the derivative
    # preserves the norm to first order.
    rng = np.random.default_rng(6)
    for _ in range(10):
        q = _random_unit_quaternion(rng)
        omega = rng.standard_normal(3)
        assert np.isclose(np.dot(q, quaternion_derivative(q, omega)), 0.0, atol=1e-12)


def test_derivative_matches_finite_difference():
    # Integrating p_dot for a constant world-frame omega must match the
    # axis-angle quaternion of the accumulated rotation.
    omega = np.array([0.0, 0.0, 1.0])
    dt = 1e-6
    q = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.3)
    q_next = quaternion_normalize(q + dt * quaternion_derivative(q, omega))
    q_expected = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.3 + dt)
    assert np.allclose(q_next, q_expected, atol=1e-9)
