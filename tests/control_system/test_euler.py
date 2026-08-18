import numpy as np
import pytest

from simulations.control_system.euler import (
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    skew,
    unskew,
    wrap_to_pi,
)


def test_identity_rotation():
    np.testing.assert_allclose(euler_to_rotation_matrix(0.0, 0.0, 0.0), np.eye(3), atol=1e-12)


def test_rotation_matrix_is_orthonormal_and_right_handed():
    R = euler_to_rotation_matrix(0.3, -0.7, 1.9)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_yaw_rotates_body_x_into_world_y():
    """A 90 deg yaw must map body +x onto world +y (right-handed, z up)."""
    R = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2.0)
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_roll_rotates_body_y_towards_world_z():
    R = euler_to_rotation_matrix(np.pi / 2.0, 0.0, 0.0)
    np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)


def test_pitch_tilts_body_z_towards_world_x():
    R = euler_to_rotation_matrix(0.0, np.pi / 2.0, 0.0)
    np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-12)


def test_zyx_composition_order():
    """R must be R_z R_y R_x, not any other ordering of the same factors."""
    roll, pitch, yaw = 0.4, -0.2, 1.1
    R_x = euler_to_rotation_matrix(roll, 0.0, 0.0)
    R_y = euler_to_rotation_matrix(0.0, pitch, 0.0)
    R_z = euler_to_rotation_matrix(0.0, 0.0, yaw)
    np.testing.assert_allclose(
        euler_to_rotation_matrix(roll, pitch, yaw), R_z @ R_y @ R_x, atol=1e-12
    )


@pytest.mark.parametrize(
    "angles",
    [
        (0.0, 0.0, 0.0),
        (0.3, 0.2, -0.5),
        (-1.2, 0.9, 2.8),
        (np.pi - 0.01, 0.0, -np.pi + 0.01),
        (0.7, -1.4, 0.0),
    ],
)
def test_euler_roundtrip(angles):
    R = euler_to_rotation_matrix(*angles)
    np.testing.assert_allclose(rotation_matrix_to_euler(R), angles, atol=1e-9)


@pytest.mark.parametrize("pitch", [np.pi / 2.0, -np.pi / 2.0])
def test_gimbal_lock_recovers_the_rotation(pitch):
    """At +/-90 deg pitch, roll and yaw are degenerate but R must round-trip."""
    R = euler_to_rotation_matrix(0.6, pitch, -0.9)
    recovered = rotation_matrix_to_euler(R)
    assert recovered[0] == 0.0
    assert recovered[1] == pytest.approx(pitch)
    np.testing.assert_allclose(euler_to_rotation_matrix(*recovered), R, atol=1e-7)


def test_rotation_matrix_to_euler_rejects_bad_shape():
    with pytest.raises(ValueError):
        rotation_matrix_to_euler(np.eye(4))


def test_skew_matches_cross_product():
    v = np.array([0.3, -1.2, 0.8])
    w = np.array([2.0, 0.5, -0.7])
    np.testing.assert_allclose(skew(v) @ w, np.cross(v, w), atol=1e-12)


def test_skew_is_antisymmetric():
    S = skew([1.0, 2.0, 3.0])
    np.testing.assert_allclose(S, -S.T, atol=1e-12)


def test_unskew_inverts_skew():
    v = np.array([0.4, -0.2, 1.7])
    np.testing.assert_allclose(unskew(skew(v)), v, atol=1e-12)


def test_unskew_reads_the_vee_components_of_the_spec():
    """(.)^v must pick out [S32, S13, S21] as written in the attitude section."""
    S = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    np.testing.assert_allclose(unskew(S), [S[2, 1], S[0, 2], S[1, 0]], atol=1e-12)


def test_unskew_of_an_antisymmetrised_matrix():
    """The controller's 0.5*(M - M.T) form must drop any symmetric part."""
    v = np.array([0.4, -0.2, 1.7])
    symmetric = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    M = skew(v) + symmetric
    np.testing.assert_allclose(0.5 * unskew(M - M.T), v, atol=1e-12)


def test_wrap_to_pi():
    np.testing.assert_allclose(
        wrap_to_pi([0.0, 3.0 * np.pi, -3.0 * np.pi, 0.5, np.pi + 0.1]),
        [0.0, -np.pi, -np.pi, 0.5, -np.pi + 0.1],
        atol=1e-12,
    )
