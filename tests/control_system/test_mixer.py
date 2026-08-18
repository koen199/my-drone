import numpy as np
import pytest

from simulations.control_system.mixer import Mixer


def test_allocation_matrix_matches_the_spec_equations(geometry):
    """Rows must reproduce the F/Tx/Ty/Tz equations of the mixer section.

    The Tz row follows the equation in the spec, not the printed allocation
    matrix: that matrix still carries the pre-correction [c, -c, c, -c] and
    contradicts its own Tz equation.
    """
    l1, l2 = geometry.roll_arm, geometry.pitch_arm
    c = geometry.drag_to_thrust_ratio
    np.testing.assert_allclose(
        Mixer(geometry).allocation_matrix,
        [
            [1.0, 1.0, 1.0, 1.0],
            [-l1, -l1, +l1, +l1],
            [-l2, +l2, +l2, -l2],
            [-c, +c, -c, +c],
        ],
    )


def test_allocation_rows_are_orthogonal(geometry):
    """Orthogonal rows are what makes A invertible for any positive geometry."""
    A = Mixer(geometry).allocation_matrix
    gram = A @ A.T
    np.testing.assert_allclose(gram, np.diag(np.diag(gram)), atol=1e-12)
    assert np.linalg.matrix_rank(A) == 4


@pytest.mark.parametrize(
    "effort",
    [
        (10.0, 0.0, 0.0, 0.0),
        (12.0, 0.4, -0.3, 0.05),
        (9.0, -0.2, 0.2, -0.02),
    ],
)
def test_allocation_roundtrip(geometry, effort):
    """Unsaturated, the achieved wrench must equal the requested one."""
    mixer = Mixer(geometry)
    thrust, *torque = effort
    output = mixer.mix(thrust, np.array(torque))
    assert not output.saturated
    np.testing.assert_allclose(mixer.wrench(output.motor_thrusts), effort, atol=1e-9)


def test_hover_splits_thrust_evenly(geometry):
    mixer = Mixer(geometry)
    output = mixer.mix(11.772, np.zeros(3))
    np.testing.assert_allclose(output.motor_thrusts, np.full(4, 11.772 / 4.0), rtol=1e-9)


def test_positive_roll_torque_favours_the_positive_y_motors(geometry):
    """Tx > 0 needs more thrust from M3/M4 (at +l1) than from M1/M2."""
    output = Mixer(geometry).mix(12.0, np.array([0.3, 0.0, 0.0]))
    f = output.motor_thrusts
    assert f[2] > f[0] and f[2] > f[1]
    assert f[3] > f[0] and f[3] > f[1]


def test_positive_pitch_torque_favours_the_negative_x_motors(geometry):
    """Ty > 0 needs more thrust from M2/M3 (at -l2) than from M1/M4."""
    f = Mixer(geometry).mix(12.0, np.array([0.0, 0.3, 0.0])).motor_thrusts
    assert f[1] > f[0] and f[1] > f[3]
    assert f[2] > f[0] and f[2] > f[3]


def test_positive_yaw_torque_favours_the_clockwise_motors(geometry):
    """Reaction opposes spin, so +Tz comes from speeding up the CW pair M2/M4."""
    f = Mixer(geometry).mix(12.0, np.array([0.0, 0.0, 0.05])).motor_thrusts
    assert f[1] > f[0] and f[1] > f[2]
    assert f[3] > f[0] and f[3] > f[2]


def test_pure_torque_does_not_change_total_thrust(geometry):
    """The torque rows are orthogonal to the thrust row, so F is untouched."""
    f = Mixer(geometry).mix(12.0, np.array([0.3, -0.2, 0.04])).motor_thrusts
    assert np.sum(f) == pytest.approx(12.0)


def test_tilt_compensation(geometry):
    mixer = Mixer(geometry)
    roll, pitch = 0.3, -0.2
    assert mixer.total_thrust(10.0, roll, pitch) == pytest.approx(
        10.0 / (np.cos(roll) * np.cos(pitch))
    )
    assert mixer.total_thrust(10.0, 0.0, 0.0) == pytest.approx(10.0)


def test_tilt_compensation_is_bounded_at_extreme_attitudes(geometry):
    """cos(phi)cos(theta) -> 0 at 90 deg must not blow the command up."""
    mixer = Mixer(geometry, min_tilt_cosine=0.5)
    assert mixer.total_thrust(10.0, np.pi / 2.0, 0.0) == pytest.approx(20.0)


def test_saturation_is_reported_and_clipped(geometry):
    mixer = Mixer(geometry)
    output = mixer.mix(1e4, np.zeros(3))
    assert output.saturated
    assert np.all(output.motor_speeds <= geometry.max_motor_speed + 1e-12)
    assert np.all(output.motor_thrusts <= geometry.max_motor_thrust + 1e-9)


def test_negative_thrust_requests_are_clipped_to_idle(geometry):
    """A torque larger than the available thrust would need a pulling rotor."""
    output = mixer_output = Mixer(geometry).mix(1.0, np.array([5.0, 0.0, 0.0]))
    assert output.saturated
    assert np.min(mixer_output.requested_thrusts) < 0.0
    assert np.all(output.motor_thrusts >= geometry.min_motor_thrust - 1e-12)


def test_motor_speeds_are_consistent_with_the_thrusts(geometry):
    output = Mixer(geometry).mix(12.0, np.array([0.2, -0.1, 0.03]))
    np.testing.assert_allclose(
        geometry.speed_to_thrust(output.motor_speeds), output.motor_thrusts, rtol=1e-12
    )


def test_invalid_arguments(geometry):
    with pytest.raises(ValueError):
        Mixer(geometry, min_tilt_cosine=0.0)
    with pytest.raises(ValueError):
        Mixer(geometry, min_tilt_cosine=1.5)
    with pytest.raises(ValueError):
        Mixer(geometry).mix(10.0, np.zeros(2))
