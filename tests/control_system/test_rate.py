import numpy as np
import pytest

from simulations.control_system.rate import (
    GRAVITY,
    AngularRateController,
    VerticalSpeedController,
)


# ----------------------------------------------------------------------
# Angular rate controller
# ----------------------------------------------------------------------


def test_torque_is_inertia_times_commanded_acceleration(inertia):
    """Kp = 1 /s and an error of 1 rad/s must ask for 1 rad/s^2, i.e. T = I."""
    controller = AngularRateController(inertia, kp=1.0)
    torque = controller.update([1.0, 1.0, 1.0], np.zeros(3), dt=0.01)
    np.testing.assert_allclose(torque, inertia @ np.ones(3), atol=1e-12)


def test_zero_error_gives_zero_torque(inertia):
    controller = AngularRateController(inertia, kp=10.0, ki=5.0, kd=1.0)
    torque = controller.update([0.3, -0.2, 0.1], [0.3, -0.2, 0.1], dt=0.01)
    np.testing.assert_allclose(torque, np.zeros(3), atol=1e-12)


def test_torque_opposes_the_rate_error_sign(inertia):
    controller = AngularRateController(inertia, kp=5.0)
    torque = controller.update(np.zeros(3), [1.0, -1.0, 2.0], dt=0.01)
    assert torque[0] < 0.0 and torque[1] > 0.0 and torque[2] < 0.0


def test_axes_are_decoupled(inertia):
    """Diagonal gains and a diagonal inertia must not cross-couple axes."""
    controller = AngularRateController(inertia, kp=[1.0, 2.0, 3.0], ki=1.0, kd=0.1)
    torque = controller.update([1.0, 0.0, 0.0], np.zeros(3), dt=0.01)
    assert torque[1] == pytest.approx(0.0)
    assert torque[2] == pytest.approx(0.0)


def test_integral_builds_torque_against_a_persistent_error(inertia):
    controller = AngularRateController(inertia, kp=0.0, ki=1.0)
    torques = [controller.update(np.ones(3), np.zeros(3), dt=0.01)[0] for _ in range(10)]
    assert torques[-1] > torques[0] > 0.0
    assert torques[-1] == pytest.approx(inertia[0, 0] * 1.0 * 0.1)


def test_integral_limit_is_in_angular_acceleration_units(inertia):
    controller = AngularRateController(inertia, kp=0.0, ki=100.0, integral_limit=2.0)
    for _ in range(100):
        torque = controller.update(np.ones(3), np.zeros(3), dt=0.01)
    np.testing.assert_allclose(torque, inertia @ (2.0 * np.ones(3)), atol=1e-9)


def test_max_angular_acceleration_clamps_the_torque(inertia):
    controller = AngularRateController(inertia, kp=1000.0, max_angular_acceleration=30.0)
    torque = controller.update(np.ones(3), np.zeros(3), dt=0.01)
    np.testing.assert_allclose(torque, inertia @ (30.0 * np.ones(3)), atol=1e-12)


def test_reset_clears_the_integral(inertia):
    controller = AngularRateController(inertia, kp=0.0, ki=1.0)
    for _ in range(10):
        controller.update(np.ones(3), np.zeros(3), dt=0.01)
    controller.reset()
    np.testing.assert_allclose(
        controller.update(np.zeros(3), np.zeros(3), dt=0.01), np.zeros(3), atol=1e-12
    )


def test_rate_controller_rejects_bad_inertia():
    with pytest.raises(ValueError):
        AngularRateController(np.eye(2), kp=1.0)


# ----------------------------------------------------------------------
# Vertical speed controller
# ----------------------------------------------------------------------


def test_integral_is_preloaded_with_a_fraction_of_hover_thrust(mass):
    """At zero error the output must already be 0.8 of the hover force."""
    controller = VerticalSpeedController(mass, kp=1.0, hover_integral_fraction=0.8)
    force = controller.update(0.0, 0.0, dt=0.01)
    assert force == pytest.approx(0.8 * mass * GRAVITY)


def test_no_preload_gives_zero_force_at_zero_error(mass):
    controller = VerticalSpeedController(mass, kp=1.0, hover_integral_fraction=0.0)
    assert controller.update(0.0, 0.0, dt=0.01) == pytest.approx(0.0)


def test_force_is_mass_times_commanded_acceleration(mass):
    controller = VerticalSpeedController(mass, kp=3.0, hover_integral_fraction=0.0)
    assert controller.update(2.0, 0.0, dt=0.01) == pytest.approx(mass * 3.0 * 2.0)


def test_integral_winds_up_to_the_remaining_hover_force(mass):
    """Holding a hover, the integrator must supply the last 20 % of m*g."""
    controller = VerticalSpeedController(
        mass, kp=0.0, ki=5.0, hover_integral_fraction=0.8
    )
    dt = 0.01
    # A steady 0.2*g/5 m/s error integrated for 1 s adds exactly 0.2*g.
    error = 0.2 * GRAVITY / 5.0
    for _ in range(100):
        force = controller.update(error, 0.0, dt)
    assert force == pytest.approx(mass * GRAVITY, rel=1e-9)


def test_output_never_asks_the_propellers_to_pull(mass):
    """F_z is clamped at zero: fixed-pitch rotors cannot produce down-force."""
    controller = VerticalSpeedController(mass, kp=10.0, hover_integral_fraction=0.0)
    assert controller.update(-100.0, 0.0, dt=0.01) == pytest.approx(0.0)


def test_max_vertical_acceleration_clamps_the_force(mass):
    controller = VerticalSpeedController(
        mass, kp=100.0, hover_integral_fraction=0.0, max_vertical_acceleration=20.0
    )
    assert controller.update(10.0, 0.0, dt=0.01) == pytest.approx(mass * 20.0)


def test_reset_restores_the_preload(mass):
    controller = VerticalSpeedController(mass, kp=1.0, ki=10.0, hover_integral_fraction=0.8)
    for _ in range(50):
        controller.update(1.0, 0.0, dt=0.01)
    controller.reset()
    assert controller.update(0.0, 0.0, dt=0.01) == pytest.approx(0.8 * mass * GRAVITY)


def test_vertical_controller_rejects_bad_mass():
    with pytest.raises(ValueError):
        VerticalSpeedController(mass=0.0, kp=1.0)
