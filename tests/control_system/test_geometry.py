import numpy as np
import pytest

from simulations.control_system.geometry import QuadrotorGeometry


def test_motor_layout_matches_the_mixer_figure(geometry):
    """M1 (+x,-y), M2 (-x,-y), M3 (-x,+y), M4 (+x,+y); l1 is the y offset."""
    l1, l2 = geometry.roll_arm, geometry.pitch_arm
    np.testing.assert_allclose(
        geometry.motor_positions,
        [[+l2, -l1, 0.0], [-l2, -l1, 0.0], [-l2, +l1, 0.0], [+l2, +l1, 0.0]],
    )


def test_motors_one_and_three_spin_counterclockwise(geometry):
    np.testing.assert_allclose(geometry.spin_directions, [1.0, -1.0, 1.0, -1.0])


def test_thrust_speed_roundtrip(geometry):
    speeds = np.array([100.0, 400.0, 700.0, 1000.0])
    np.testing.assert_allclose(
        geometry.thrust_to_speed(geometry.speed_to_thrust(speeds)), speeds, rtol=1e-12
    )


def test_thrust_follows_the_square_law(geometry):
    assert geometry.speed_to_thrust(700.0) == pytest.approx(
        geometry.thrust_coefficient * 700.0 ** 2
    )
    # Doubling the speed quadruples the thrust.
    assert geometry.speed_to_thrust(400.0) == pytest.approx(
        4.0 * geometry.speed_to_thrust(200.0)
    )


def test_thrust_to_speed_clips_to_the_actuator_range(geometry):
    assert geometry.thrust_to_speed(1e6) == pytest.approx(geometry.max_motor_speed)
    assert geometry.thrust_to_speed(-5.0) == pytest.approx(geometry.min_motor_speed)


def test_drag_to_thrust_ratio(geometry):
    assert geometry.drag_to_thrust_ratio == pytest.approx(
        geometry.drag_coefficient / geometry.thrust_coefficient
    )


def test_reaction_torque_opposes_the_spin(geometry):
    """A CCW motor must drag the airframe clockwise, and vice versa."""
    reaction = geometry.reaction_torque(np.full(4, 500.0))
    np.testing.assert_allclose(np.sign(reaction), -geometry.spin_directions)


def test_reaction_torque_magnitude(geometry):
    reaction = geometry.reaction_torque(np.full(4, 500.0))
    np.testing.assert_allclose(
        np.abs(reaction), geometry.drag_coefficient * 500.0 ** 2, rtol=1e-12
    )


def test_reaction_torque_per_thrust_is_the_c_of_the_spec(geometry):
    np.testing.assert_allclose(
        np.abs(geometry.reaction_torque_per_thrust),
        geometry.drag_to_thrust_ratio,
        rtol=1e-12,
    )


def test_equal_and_opposite_pairs_cancel_in_yaw(geometry):
    """All four motors at the same speed must produce no net yaw torque."""
    assert np.sum(geometry.reaction_torque(np.full(4, 650.0))) == pytest.approx(0.0)


def test_thrust_limits(geometry):
    assert geometry.max_motor_thrust == pytest.approx(
        geometry.thrust_coefficient * geometry.max_motor_speed ** 2
    )
    assert geometry.min_motor_thrust == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"roll_arm": 0.0},
        {"pitch_arm": -0.1},
        {"thrust_coefficient": 0.0},
        {"drag_coefficient": -1e-8},
        {"min_motor_speed": -1.0},
        {"max_motor_speed": 0.0},
    ],
)
def test_invalid_arguments(kwargs):
    valid = {
        "roll_arm": 0.17,
        "pitch_arm": 0.13,
        "thrust_coefficient": 6e-6,
        "drag_coefficient": 9.6e-8,
        "min_motor_speed": 0.0,
        "max_motor_speed": 1200.0,
    }
    with pytest.raises(ValueError):
        QuadrotorGeometry(**{**valid, **kwargs})
