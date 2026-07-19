import numpy as np
import pytest
from simulations.ahrs.math_utils import (
    safe_normalize,
    skew_symmetric,
    gram_schmidt,
    rodrigues_rotation,
    rodrigues_small_angle,
)


def test_safe_normalize_unit_length():
    v = np.array([3.0, 4.0, 0.0])
    assert np.isclose(np.linalg.norm(safe_normalize(v)), 1.0)


def test_safe_normalize_raises_on_zero():
    with pytest.raises(ValueError):
        safe_normalize(np.zeros(3))


def test_skew_symmetric_shape():
    assert skew_symmetric(np.ones(3)).shape == (3, 3)


def test_skew_symmetric_antisymmetric():
    omega = np.array([1.0, 2.0, 3.0])
    O = skew_symmetric(omega)
    assert np.allclose(O + O.T, 0.0)


def test_skew_symmetric_cross_product_equivalence():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        assert np.allclose(skew_symmetric(a) @ b, np.cross(a, b))


def test_gram_schmidt_orthogonality():
    rng = np.random.default_rng(1)
    R = np.eye(3) + 0.1 * rng.standard_normal((3, 3))
    R_orth = gram_schmidt(R)
    assert np.allclose(R_orth @ R_orth.T, np.eye(3), atol=1e-10)


def test_gram_schmidt_determinant():
    rng = np.random.default_rng(2)
    R = np.eye(3) + 0.1 * rng.standard_normal((3, 3))
    assert np.isclose(np.linalg.det(gram_schmidt(R)), 1.0, atol=1e-10)


def test_gram_schmidt_preserves_valid_matrix():
    # Build a known valid rotation matrix (90 deg about x, then 45 deg about z)
    cx, sx = np.cos(np.pi / 2), np.sin(np.pi / 2)
    cz, sz = np.cos(np.pi / 4), np.sin(np.pi / 4)
    Rx = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    Rz = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    R = Rx @ Rz
    assert np.allclose(gram_schmidt(R), R, atol=1e-10)


def test_rodrigues_rotation_identity():
    axis = np.array([0.0, 0.0, 1.0])
    assert np.allclose(rodrigues_rotation(axis, 0.0), np.eye(3))


def test_rodrigues_rotation_90_deg_z():
    axis = np.array([0.0, 0.0, 1.0])
    R = rodrigues_rotation(axis, np.pi / 2)
    rotated = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-10)


def test_rodrigues_rotation_is_orthogonal():
    rng = np.random.default_rng(3)
    axis = safe_normalize(rng.standard_normal(3))
    R = rodrigues_rotation(axis, 1.23)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_rodrigues_small_angle_limit():
    axis = np.array([0.0, 1.0, 0.0])
    angle = 1e-5
    R_full = rodrigues_rotation(axis, angle)
    R_small = rodrigues_small_angle(axis, angle)
    assert np.allclose(R_full, R_small, atol=1e-9)
