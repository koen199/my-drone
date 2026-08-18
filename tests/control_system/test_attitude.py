import numpy as np
import pytest

from simulations.control_system.attitude import AttitudeController, orientation_error
from simulations.control_system.euler import euler_to_rotation_matrix


def test_zero_error_when_aligned():
    R = euler_to_rotation_matrix(0.3, -0.2, 1.4)
    np.testing.assert_allclose(orientation_error(R, R), np.zeros(3), atol=1e-12)


def test_error_is_well_defined_at_the_singularity():
    """theta -> 0 is where the axis-angle form divides by zero; e_R must not."""
    R_m = np.eye(3)
    R_d = euler_to_rotation_matrix(1e-12, 0.0, 0.0)
    error = orientation_error(R_m, R_d)
    assert np.all(np.isfinite(error))
    np.testing.assert_allclose(error, np.zeros(3), atol=1e-11)


def test_error_equals_sin_theta_times_axis():
    """e_R must reproduce sin(theta) * u for a known axis-angle rotation."""
    axis = np.array([1.0, -2.0, 0.5])
    axis = axis / np.linalg.norm(axis)
    theta = 0.7
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    R_e = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    R_m = euler_to_rotation_matrix(0.2, 0.4, -0.9)
    R_d = R_m @ R_e
    np.testing.assert_allclose(orientation_error(R_m, R_d), np.sin(theta) * axis, atol=1e-12)


def test_error_is_first_order_equal_to_theta_times_axis():
    """For small errors sin(theta) ~ theta, so e_R ~ theta * u as the spec says."""
    theta = 1e-3
    R_m = np.eye(3)
    R_d = euler_to_rotation_matrix(0.0, 0.0, theta)
    np.testing.assert_allclose(orientation_error(R_m, R_d), [0.0, 0.0, theta], rtol=1e-6)


def test_error_is_expressed_in_the_body_frame():
    """A yaw-rotated vehicle rolling to level must see a pure body-x error."""
    R_m = euler_to_rotation_matrix(0.1, 0.0, np.pi / 2.0)
    R_d = euler_to_rotation_matrix(0.0, 0.0, np.pi / 2.0)
    error = orientation_error(R_m, R_d)
    np.testing.assert_allclose(error[1:], [0.0, 0.0], atol=1e-12)
    assert error[0] == pytest.approx(-np.sin(0.1))


def test_error_sign_drives_towards_the_setpoint():
    """Applying omega_sp for a short time must reduce the attitude error."""
    controller = AttitudeController(kp=10.0)
    R_m = euler_to_rotation_matrix(0.0, 0.0, 0.0)
    R_d = euler_to_rotation_matrix(0.25, -0.15, 0.35)

    dt = 1e-3
    previous = np.linalg.norm(orientation_error(R_m, R_d))
    for _ in range(2000):
        omega = controller.update(R_m, R_d)
        # Integrate R_m forward with body-frame omega: R <- R (I + [w]x dt).
        W = np.array([
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ])
        R_m = R_m @ (np.eye(3) + W * dt)
        # Re-orthonormalise so the first-order update does not drift.
        u, _, vt = np.linalg.svd(R_m)
        R_m = u @ vt

        # The error must shrink on every single step, never oscillate.
        current = np.linalg.norm(orientation_error(R_m, R_d))
        assert current < previous
        previous = current

    assert previous < 1e-6


def test_proportional_gain_scales_the_rate_setpoint():
    R_m = np.eye(3)
    R_d = euler_to_rotation_matrix(0.0, 0.0, 0.2)
    slow = AttitudeController(kp=1.0).update(R_m, R_d)
    fast = AttitudeController(kp=5.0).update(R_m, R_d)
    np.testing.assert_allclose(fast, 5.0 * slow, atol=1e-12)


def test_per_axis_gain():
    R_m = np.eye(3)
    R_d = euler_to_rotation_matrix(0.1, 0.1, 0.1)
    uniform = AttitudeController(kp=1.0).update(R_m, R_d)
    per_axis = AttitudeController(kp=[1.0, 2.0, 3.0]).update(R_m, R_d)
    np.testing.assert_allclose(per_axis, uniform * np.array([1.0, 2.0, 3.0]), atol=1e-12)


def test_rate_saturation():
    controller = AttitudeController(kp=100.0, max_rate=[1.0, 1.0, 0.5])
    rate = controller.update(np.eye(3), euler_to_rotation_matrix(0.5, -0.5, 0.5))
    assert np.all(np.abs(rate) <= np.array([1.0, 1.0, 0.5]) + 1e-12)
    assert rate[0] == pytest.approx(1.0)
    assert rate[1] == pytest.approx(-1.0)


def test_update_from_euler_matches_update():
    controller = AttitudeController(kp=2.0)
    measured = (0.1, -0.2, 0.3)
    desired = (0.0, 0.1, -0.4)
    np.testing.assert_allclose(
        controller.update_from_euler(measured, desired),
        controller.update(
            euler_to_rotation_matrix(*measured), euler_to_rotation_matrix(*desired)
        ),
        atol=1e-12,
    )


def test_invalid_arguments():
    with pytest.raises(ValueError):
        AttitudeController(kp=[1.0, 2.0])
    with pytest.raises(ValueError):
        AttitudeController(kp=-1.0)
    with pytest.raises(ValueError):
        orientation_error(np.eye(2), np.eye(3))
