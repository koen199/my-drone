"""Propeller plant model tests, including the handshake with the rigid body."""
import numpy as np
import pytest

from simulations.control_system.mixer import Mixer
from simulations.control_system.propellers import Propellers
from simulations.rigid_body_simulator.frames import CoordinateSystem, Vector
from simulations.rigid_body_simulator.rigid_body import RigidBody


def _body_vector(components) -> Vector:
    return Vector(components=components, coordinate_system=CoordinateSystem.BODY)


def _net_wrench(drone: RigidBody, mass: float, inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the total force and CoM torque the rigid body currently sees.

    Only valid at identity attitude with zero angular velocity, where the
    world and body frames coincide and the gyroscopic term vanishes, so
    F = m*a and T = I*alpha directly.
    """
    derivative = drone.compute_state_derivative(0.0, drone.get_state())
    return mass * derivative[3:6], inertia @ derivative[10:13]


def test_speeds_are_clipped_to_the_actuator_range(geometry):
    propellers = Propellers(geometry)
    propellers.set_speed([-100.0, 0.0, 5000.0, 700.0])
    np.testing.assert_allclose(
        propellers.speeds, [0.0, 0.0, geometry.max_motor_speed, 700.0]
    )


def test_thrust_acts_along_body_z_at_the_motor_positions(geometry):
    loads = Propellers(geometry).set_speed(np.full(4, 700.0))
    thrust = geometry.speed_to_thrust(700.0)
    assert len(loads.forces) == 4
    for force, position in zip(loads.forces, geometry.motor_positions):
        assert force.force.coordinate_system is CoordinateSystem.BODY
        np.testing.assert_allclose(force.force.components, [0.0, 0.0, thrust])
        np.testing.assert_allclose(force.application_point.components, position)


def test_application_points_are_shifted_by_the_cog_offset(geometry):
    """RigidBody wants body-origin-relative points; geometry is CoG-relative."""
    cog = np.array([0.02, -0.01, 0.05])
    loads = Propellers(geometry, cog_position=_body_vector(cog)).set_speed(np.full(4, 500.0))
    for force, position in zip(loads.forces, geometry.motor_positions):
        np.testing.assert_allclose(force.application_point.components, position + cog)


def test_reaction_torque_is_a_single_pure_torque_about_body_z(geometry):
    loads = Propellers(geometry).set_speed([700.0, 500.0, 700.0, 500.0])
    assert len(loads.torques) == 1
    torque = loads.torques[0].torque
    assert torque.coordinate_system is CoordinateSystem.BODY
    np.testing.assert_allclose(torque.components[:2], [0.0, 0.0])
    expected = np.sum(geometry.reaction_torque(np.array([700.0, 500.0, 700.0, 500.0])))
    assert torque.components[2] == pytest.approx(expected)


def test_balanced_propellers_produce_no_yaw_torque(geometry):
    loads = Propellers(geometry).set_speed(np.full(4, 700.0))
    assert loads.torques[0].torque.components[2] == pytest.approx(0.0)


def test_cog_position_must_be_in_the_body_frame(geometry):
    with pytest.raises(ValueError):
        Propellers(
            geometry,
            cog_position=Vector(components=[0.0, 0.0, 0.0], coordinate_system=CoordinateSystem.WORLD),
        )


def test_set_speed_rejects_bad_shape(geometry):
    with pytest.raises(ValueError):
        Propellers(geometry).set_speed([1.0, 2.0, 3.0])


# ----------------------------------------------------------------------
# Mixer <-> propellers <-> rigid body consistency
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cog", [(0.0, 0.0, 0.0), (0.03, -0.02, 0.04)])
@pytest.mark.parametrize(
    "effort",
    [
        (11.772, 0.0, 0.0, 0.0),
        (12.0, 0.25, -0.18, 0.04),
        (9.0, -0.1, 0.15, -0.03),
    ],
)
def test_mixer_request_is_what_the_rigid_body_actually_feels(
    geometry, mass, inertia, cog, effort
):
    """The whole allocation chain must be self-consistent.

    Ask the mixer for a wrench, convert to motor speeds, spin the propellers,
    hand the resulting loads to the rigid body, and read the net force and
    torque back out of its state derivative. Any sign error in the motor
    layout, the reaction torque, or the CoG bookkeeping shows up here.
    """
    thrust, *torque = effort
    mixer = Mixer(geometry)
    propellers = Propellers(geometry, cog_position=Vector(
        components=cog, coordinate_system=CoordinateSystem.BODY
    ))
    drone = RigidBody(
        inertia_body=inertia,
        mass=mass,
        cog_position=Vector(components=cog, coordinate_system=CoordinateSystem.BODY),
    )

    output = mixer.mix(thrust, np.array(torque))
    assert not output.saturated

    loads = propellers.set_speed(output.motor_speeds)
    drone.apply_forces(loads.forces)
    drone.apply_torques(loads.torques)

    force_total, torque_total = _net_wrench(drone, mass, inertia)
    np.testing.assert_allclose(force_total, [0.0, 0.0, thrust], atol=1e-8)
    np.testing.assert_allclose(torque_total, torque, atol=1e-8)


def test_yaw_torque_sign_survives_the_round_trip(geometry, mass, inertia):
    """A positive Tz request must yaw the body positively, not negatively."""
    mixer = Mixer(geometry)
    propellers = Propellers(geometry)
    drone = RigidBody(
        inertia_body=inertia, mass=mass, cog_position=_body_vector([0.0, 0.0, 0.0])
    )

    output = mixer.mix(12.0, np.array([0.0, 0.0, 0.05]))
    loads = propellers.set_speed(output.motor_speeds)
    drone.apply_forces(loads.forces)
    drone.apply_torques(loads.torques)

    _, torque_total = _net_wrench(drone, mass, inertia)
    assert torque_total[2] > 0.0
