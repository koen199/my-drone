"""
Closed-loop tests: the control system flying the rigid body simulator.

These are the tests that prove the pieces fit together. Each one builds the
full loop (controller -> mixer -> propellers -> RigidBody -> RK4 -> state
estimate) and asserts on the trajectory rather than on any single call.
"""
import numpy as np
import pytest

from simulations.control_system.control_system import (
    ControlGains,
    ControlSystem,
    Setpoint,
    StateEstimate,
)
from simulations.control_system.euler import rotation_matrix_to_euler
from simulations.control_system.propellers import Propellers
from simulations.rigid_body_simulator.frames import CoordinateSystem, Force, Vector
from simulations.rigid_body_simulator.rigid_body import RigidBody
from simulations.rigid_body_simulator.solvers import RK4

GRAVITY = 9.81
DT = 0.004
CONTROL_DECIMATION = 1  # 250 Hz control, matching the physics rate

# 250 Hz is well inside RK4's accuracy for a plant whose fastest pole is the
# 40 rad/s rate loop: raising the physics to 1 kHz with the control still at
# 250 Hz reproduces these trajectories to five decimals, but runs 3.5x slower.
# The demo script uses the 1 kHz / 250 Hz split.


def _measure(drone: RigidBody) -> StateEstimate:
    """Perfect state estimator: read the truth straight off the rigid body."""
    roll, pitch, yaw = rotation_matrix_to_euler(drone.rotation_matrix)
    body_rates = drone.get_rotation_speed(
        Vector(components=[0.0, 0.0, 0.0], coordinate_system=CoordinateSystem.BODY)
    ).components
    return StateEstimate(
        altitude=drone.position[2],
        vertical_speed=drone.velocity[2],
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        body_rates=body_rates,
    )


def _fly(
    geometry,
    mass,
    inertia,
    setpoint: Setpoint,
    duration: float,
    gains: ControlGains | None = None,
    initial_attitude: np.ndarray | None = None,
    initial_altitude: float = 0.0,
    cog=(0.0, 0.0, 0.0),
):
    """Run the closed loop and return the per-control-step history."""
    cog_vector = Vector(components=cog, coordinate_system=CoordinateSystem.BODY)
    drone = RigidBody(
        inertia_body=inertia,
        mass=mass,
        cog_position=cog_vector,
        initial_position=Vector(
            components=[0.0, 0.0, initial_altitude - cog[2]],
            coordinate_system=CoordinateSystem.WORLD,
        ),
        initial_orientation=initial_attitude,
    )
    propellers = Propellers(geometry, cog_position=cog_vector)
    control_system = ControlSystem(mass, inertia, geometry, gains=gains)

    gravity = Force(
        application_point=cog_vector,
        force=Vector(
            components=[0.0, 0.0, -mass * GRAVITY], coordinate_system=CoordinateSystem.WORLD
        ),
    )
    solver = RK4()
    solver.register(drone)

    history = {"time": [], "altitude": [], "attitude": [], "speeds": []}
    control_dt = DT * CONTROL_DECIMATION

    for step in range(int(duration / DT)):
        if step % CONTROL_DECIMATION == 0:
            estimate = _measure(drone)
            output = control_system.update(setpoint, estimate, control_dt)
            loads = propellers.set_speed(output.motor_speeds)
            drone.apply_forces(loads.forces + [gravity])
            drone.apply_torques(loads.torques)

            history["time"].append(solver.time)
            history["altitude"].append(estimate.altitude)
            history["attitude"].append([estimate.roll, estimate.pitch, estimate.yaw])
            history["speeds"].append(output.motor_speeds)

        solver.step(DT)

    return {key: np.asarray(value) for key, value in history.items()}


def test_hover_settles_back_to_the_commanded_altitude(geometry, mass, inertia):
    """Holding 0 m from rest, the drone dips then returns, staying level.

    The dip is expected: the integrator is preloaded with only 0.8 m g, so the
    remaining 20 % of the hover force has to be integrated up while the
    vehicle sinks. See test_hover_preload_removes_the_takeoff_dip.
    """
    history = _fly(geometry, mass, inertia, Setpoint(altitude=0.0), duration=20.0)
    assert np.min(history["altitude"]) > -0.3
    assert abs(history["altitude"][-1]) < 1e-3
    assert np.max(np.abs(history["attitude"])) < 1e-6


def test_hover_preload_removes_the_takeoff_dip(geometry, mass, inertia):
    """A full 1.0 m g preload holds altitude from the first step."""
    dipped = _fly(
        geometry, mass, inertia, Setpoint(altitude=0.0), duration=5.0,
        gains=ControlGains(hover_integral_fraction=0.8),
    )
    flat = _fly(
        geometry, mass, inertia, Setpoint(altitude=0.0), duration=5.0,
        gains=ControlGains(hover_integral_fraction=1.0),
    )
    assert np.min(dipped["altitude"]) < -0.1
    assert np.min(flat["altitude"]) > -1e-6


def test_hover_motor_speeds_match_the_hover_thrust(geometry, mass, inertia):
    """In steady hover each motor must carry a quarter of the weight."""
    history = _fly(geometry, mass, inertia, Setpoint(altitude=0.0), duration=8.0)
    expected = np.sqrt((mass * GRAVITY / 4.0) / geometry.thrust_coefficient)
    np.testing.assert_allclose(history["speeds"][-1], np.full(4, expected), rtol=2e-3)


def test_altitude_step_reaches_the_setpoint(geometry, mass, inertia):
    history = _fly(geometry, mass, inertia, Setpoint(altitude=5.0), duration=15.0)
    settled = history["altitude"][history["time"] > 10.0]
    np.testing.assert_allclose(settled, 5.0, atol=0.02)


def test_altitude_step_has_no_steady_state_error(geometry, mass, inertia):
    """Zero steady-state error to a step comes from the vertical loop's integrator."""
    history = _fly(geometry, mass, inertia, Setpoint(altitude=3.0), duration=20.0)
    assert abs(history["altitude"][-1] - 3.0) < 5e-3


def test_altitude_step_does_not_overshoot_badly(geometry, mass, inertia):
    history = _fly(geometry, mass, inertia, Setpoint(altitude=2.0), duration=15.0)
    assert np.max(history["altitude"]) < 2.0 * 1.15


def test_climb_rate_respects_the_saturation_limit(geometry, mass, inertia):
    """A large step must be flown at the climb rate limit, not faster."""
    gains = ControlGains(max_climb_rate=1.5)
    history = _fly(geometry, mass, inertia, Setpoint(altitude=30.0), duration=10.0, gains=gains)
    climb_rate = np.gradient(history["altitude"], history["time"])
    assert np.max(climb_rate) < 1.5 * 1.15


@pytest.mark.parametrize(
    "setpoint",
    [
        Setpoint(roll=0.2),
        Setpoint(pitch=-0.2),
        Setpoint(yaw=0.5),
        Setpoint(roll=0.15, pitch=0.15, yaw=-0.3),
    ],
)
def test_attitude_step_is_tracked(geometry, mass, inertia, setpoint):
    history = _fly(geometry, mass, inertia, setpoint, duration=6.0)
    np.testing.assert_allclose(history["attitude"][-1], setpoint.attitude, atol=5e-3)


def test_recovers_from_a_large_initial_attitude_upset(geometry, mass, inertia):
    """Released at 40 deg roll / -30 deg pitch, the drone must level out."""
    from simulations.rigid_body_simulator.quaternion import quaternion_from_axis_angle, quaternion_multiply

    upset = quaternion_multiply(
        quaternion_from_axis_angle([0.0, 1.0, 0.0], np.deg2rad(-30.0)),
        quaternion_from_axis_angle([1.0, 0.0, 0.0], np.deg2rad(40.0)),
    )
    history = _fly(
        geometry, mass, inertia, Setpoint(altitude=0.0), duration=8.0, initial_attitude=upset
    )
    np.testing.assert_allclose(history["attitude"][-1], [0.0, 0.0, 0.0], atol=5e-3)


def test_attitude_hold_is_unaffected_by_a_cog_offset(geometry, mass, inertia):
    """The CoG offset only moves the body origin; the loop closes on the CoG."""
    history = _fly(
        geometry, mass, inertia, Setpoint(altitude=1.0), duration=10.0, cog=(0.02, -0.015, 0.03)
    )
    assert np.max(np.abs(history["attitude"])) < 1e-6
    assert abs(history["altitude"][-1] - 1.0) < 1e-2


def test_altitude_is_held_while_the_drone_is_tilted(geometry, mass, inertia):
    """Tilt compensation must keep the vertical force on target under bank."""
    history = _fly(
        geometry, mass, inertia, Setpoint(altitude=0.0, roll=0.35), duration=12.0
    )
    settled = history["altitude"][history["time"] > 8.0]
    assert np.max(np.abs(settled)) < 0.05


def test_yaw_is_held_while_rolling_and_pitching(geometry, mass, inertia):
    """Roll and pitch commands must not bleed into heading."""
    history = _fly(
        geometry, mass, inertia, Setpoint(roll=0.25, pitch=-0.25, yaw=0.0), duration=8.0
    )
    assert np.max(np.abs(history["attitude"][:, 2])) < 0.02


def test_the_loop_stays_bounded_under_aggressive_commands(geometry, mass, inertia):
    """A hard simultaneous step must not diverge even if it saturates."""
    history = _fly(
        geometry, mass, inertia, Setpoint(altitude=10.0, roll=0.6, pitch=-0.6, yaw=1.5), duration=12.0
    )
    assert np.all(np.isfinite(history["altitude"]))
    assert np.max(np.abs(history["attitude"])) < np.pi
    assert history["altitude"][-1] == pytest.approx(10.0, abs=0.1)
