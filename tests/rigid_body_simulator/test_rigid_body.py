import numpy as np
import pytest

from simulations.rigid_body_simulator.frames import CoordinateSystem, Force, Torque, Vector
from simulations.rigid_body_simulator.quaternion import (
    quaternion_from_axis_angle,
    quaternion_to_rotation_matrix,
)
from simulations.rigid_body_simulator.rigid_body import RigidBody
from simulations.rigid_body_simulator.solvers import RK4, ForwardEuler


def _rodrigues_reference(axis: np.ndarray, angle: float) -> np.ndarray:
    """Independent Rodrigues rotation matrix used as a reference in tests."""
    k = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _body_vector(components) -> Vector:
    return Vector(components=components, coordinate_system=CoordinateSystem.BODY)


def _world_vector(components) -> Vector:
    return Vector(components=components, coordinate_system=CoordinateSystem.WORLD)


def _make_body(
    inertia=None,
    mass=2.0,
    cog=(0.1, 0.2, 0.3),
    **kwargs,
) -> RigidBody:
    if inertia is None:
        inertia = np.diag([1.0, 2.0, 3.0])
    return RigidBody(
        inertia_body=inertia,
        mass=mass,
        cog_position=_body_vector(list(cog)),
        **kwargs,
    )


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------


def test_invalid_inertia_raises():
    with pytest.raises(ValueError):
        _make_body(inertia=np.ones((2, 2)))
    with pytest.raises(ValueError):
        asymmetric = np.diag([1.0, 2.0, 3.0])
        asymmetric[0, 1] = 0.5
        _make_body(inertia=asymmetric)
    with pytest.raises(ValueError):
        _make_body(inertia=np.diag([1.0, -2.0, 3.0]))


def test_invalid_mass_raises():
    with pytest.raises(ValueError):
        _make_body(mass=0.0)
    with pytest.raises(ValueError):
        _make_body(mass=-1.0)


def test_cog_must_be_body_frame():
    with pytest.raises(ValueError):
        RigidBody(
            inertia_body=np.eye(3),
            mass=1.0,
            cog_position=_world_vector([0.0, 0.0, 0.0]),
        )


def test_initial_vectors_must_be_world_frame():
    with pytest.raises(ValueError):
        _make_body(initial_position=_body_vector([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        _make_body(initial_velocity=_body_vector([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        _make_body(initial_angular_velocity=_body_vector([1.0, 0.0, 0.0]))


def test_invalid_orientation_shape_raises():
    with pytest.raises(ValueError):
        _make_body(initial_orientation=np.array([1.0, 0.0, 0.0]))


# ----------------------------------------------------------------------
# Constructor body-origin -> CoM conversion
# ----------------------------------------------------------------------


def test_initial_position_is_body_origin():
    # position property returns the CoM: origin + R_nb @ cog.
    body = _make_body(initial_position=_world_vector([1.0, 2.0, 3.0]))
    assert np.allclose(body.position, [1.1, 2.2, 3.3], atol=1e-12)


def test_initial_position_conversion_uses_orientation():
    q = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    body = _make_body(
        cog=(0.1, 0.0, 0.0),
        initial_position=_world_vector([1.0, 0.0, 0.0]),
        initial_orientation=q,
    )
    # x_hat (body) rotates onto y_hat (world).
    assert np.allclose(body.position, [1.0, 0.1, 0.0], atol=1e-10)


# ----------------------------------------------------------------------
# State interface
# ----------------------------------------------------------------------


def test_state_round_trip():
    body = _make_body(
        initial_position=_world_vector([1.0, 2.0, 3.0]),
        initial_velocity=_world_vector([0.1, 0.2, 0.3]),
        initial_angular_velocity=_world_vector([0.0, 0.0, 0.5]),
    )
    state = body.get_state()
    assert state.shape == (13,)
    body.set_state(state)
    assert np.allclose(body.get_state(), state, atol=1e-12)


def test_set_state_renormalizes_quaternion():
    body = _make_body()
    state = body.get_state()
    state[6:10] = np.array([2.0, 0.0, 0.0, 0.0])
    body.set_state(state)
    assert np.allclose(body.orientation, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_set_state_bad_shape_raises():
    body = _make_body()
    with pytest.raises(ValueError):
        body.set_state(np.zeros(12))


# ----------------------------------------------------------------------
# Forces, torques, derivative
# ----------------------------------------------------------------------


def test_force_at_cog_gives_no_angular_acceleration():
    body = _make_body()
    body.apply_forces([
        Force(application_point=_body_vector([0.1, 0.2, 0.3]), force=_world_vector([5.0, 1.0, -2.0]))
    ])
    derivative = body.compute_state_derivative(0.0, body.get_state())
    assert np.allclose(derivative[3:6], np.array([5.0, 1.0, -2.0]) / 2.0, atol=1e-12)
    assert np.allclose(derivative[10:13], np.zeros(3), atol=1e-12)


def test_offset_force_torque_hand_computed():
    body = _make_body()
    # Application point offset from the CoG by +1 body x; identity orientation.
    body.apply_forces([
        Force(application_point=_body_vector([1.1, 0.2, 0.3]), force=_world_vector([0.0, 0.0, 4.0]))
    ])
    derivative = body.compute_state_derivative(0.0, body.get_state())
    # tau = [1,0,0] x [0,0,4] = [0,-4,0]; alpha = I^-1 tau = [0,-4/2,0].
    assert np.allclose(derivative[10:13], [0.0, -2.0, 0.0], atol=1e-12)


def test_pure_torque_hand_computed():
    body = _make_body()
    body.apply_torques([Torque(torque=_world_vector([0.0, 0.0, 6.0]))])
    derivative = body.compute_state_derivative(0.0, body.get_state())
    assert np.allclose(derivative[10:13], [0.0, 0.0, 2.0], atol=1e-12)
    assert np.allclose(derivative[3:6], np.zeros(3), atol=1e-12)


def test_body_force_rotates_with_stage_quaternion():
    # A BODY-frame force must be re-expressed with the orientation of the state
    # ARGUMENT, not the stored state - this is what makes RK4 stages correct.
    body = _make_body()
    body.apply_forces([
        Force(application_point=_body_vector([0.1, 0.2, 0.3]), force=_body_vector([2.0, 0.0, 0.0]))
    ])
    state = body.get_state()
    state[6:10] = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    derivative = body.compute_state_derivative(0.0, state)
    # Body x_hat now points along world y_hat: a = [0, 2, 0] / 2.
    assert np.allclose(derivative[3:6], [0.0, 1.0, 0.0], atol=1e-10)


def test_world_application_point_is_absolute():
    body = _make_body(initial_position=_world_vector([1.0, 0.0, 0.0]))
    # CoM sits at [1.1, 0.2, 0.3]; world application point 1 m along +x from it.
    body.apply_forces([
        Force(application_point=_world_vector([2.1, 0.2, 0.3]), force=_world_vector([0.0, 0.0, 4.0]))
    ])
    derivative = body.compute_state_derivative(0.0, body.get_state())
    assert np.allclose(derivative[10:13], [0.0, -2.0, 0.0], atol=1e-12)


def test_apply_forces_replaces_previous_set():
    body = _make_body()
    cog = _body_vector([0.1, 0.2, 0.3])
    body.apply_forces([Force(application_point=cog, force=_world_vector([2.0, 0.0, 0.0]))])
    body.apply_forces([Force(application_point=cog, force=_world_vector([0.0, 2.0, 0.0]))])
    derivative = body.compute_state_derivative(0.0, body.get_state())
    assert np.allclose(derivative[3:6], [0.0, 1.0, 0.0], atol=1e-12)


def test_clear_forces_removes_forces_and_torques():
    body = _make_body()
    body.apply_forces([
        Force(application_point=_body_vector([0.0, 0.0, 0.0]), force=_world_vector([2.0, 0.0, 0.0]))
    ])
    body.apply_torques([Torque(torque=_world_vector([1.0, 0.0, 0.0]))])
    body.clear_forces()
    derivative = body.compute_state_derivative(0.0, body.get_state())
    assert np.allclose(derivative[3:13], np.zeros(10), atol=1e-12)


def test_forces_persist_across_steps():
    body = _make_body(cog=(0.0, 0.0, 0.0))
    body.apply_forces([
        Force(application_point=_body_vector([0.0, 0.0, 0.0]), force=_world_vector([2.0, 0.0, 0.0]))
    ])
    solver = RK4()
    solver.register(body)
    solver.step(0.1)
    solver.step(0.1)
    # a = 1 m/s^2 held constant: v = a * t.
    assert np.allclose(body.velocity, [0.2, 0.0, 0.0], atol=1e-10)


# ----------------------------------------------------------------------
# Sensor queries
# ----------------------------------------------------------------------


def test_acceleration_at_cog_is_force_over_mass():
    body = _make_body()
    body.apply_forces([
        Force(application_point=_body_vector([0.1, 0.2, 0.3]), force=_world_vector([4.0, 0.0, 0.0]))
    ])
    acc = body.get_acceleration(_world_vector(body.position))
    assert acc.coordinate_system is CoordinateSystem.WORLD
    assert np.allclose(acc.components, [2.0, 0.0, 0.0], atol=1e-12)


def test_acceleration_centripetal_for_pure_spin():
    omega = 2.0
    body = _make_body(
        cog=(0.0, 0.0, 0.0),
        initial_angular_velocity=_world_vector([0.0, 0.0, omega]),
    )
    rho = 0.5
    acc = body.get_acceleration(_body_vector([rho, 0.0, 0.0]))
    # Pure spin about z: a = omega x (omega x rho) = -omega^2 * rho * x_hat.
    assert acc.coordinate_system is CoordinateSystem.BODY
    assert np.allclose(acc.components, [-(omega**2) * rho, 0.0, 0.0], atol=1e-10)


def test_acceleration_output_frame_matches_input_frame():
    q = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    body = _make_body(initial_orientation=q)
    body.apply_forces([
        Force(application_point=_body_vector([0.1, 0.2, 0.3]), force=_world_vector([4.0, 0.0, 0.0]))
    ])
    acc_world = body.get_acceleration(_world_vector(body.position))
    acc_body = body.get_acceleration(_body_vector([0.1, 0.2, 0.3]))
    R_nb = quaternion_to_rotation_matrix(q)
    assert np.allclose(acc_body.components, R_nb.T @ acc_world.components, atol=1e-10)


def test_rotation_speed_frames():
    q = quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.7)
    omega_n = np.array([0.2, -0.4, 0.9])
    body = _make_body(
        initial_orientation=q,
        initial_angular_velocity=_world_vector(omega_n),
    )
    point = _body_vector([0.0, 0.55, 0.15])
    assert np.allclose(
        body.get_rotation_speed(_world_vector([0.0, 0.0, 0.0])).components, omega_n, atol=1e-12
    )
    R_nb = quaternion_to_rotation_matrix(q)
    assert np.allclose(body.get_rotation_speed(point).components, R_nb.T @ omega_n, atol=1e-10)


# ----------------------------------------------------------------------
# Integrated physics sanity
# ----------------------------------------------------------------------


def test_constant_force_gives_parabola():
    mass = 2.0
    g = 9.81
    body = _make_body(
        mass=mass,
        initial_velocity=_world_vector([1.0, 0.0, 5.0]),
    )
    r0 = body.position
    body.apply_forces([
        Force(
            application_point=_body_vector([0.1, 0.2, 0.3]),
            force=_world_vector([0.0, 0.0, -g * mass]),
        )
    ])
    solver = RK4()
    solver.register(body)
    dt, n_steps = 0.01, 100
    for _ in range(n_steps):
        solver.step(dt)
    t = dt * n_steps
    expected = r0 + np.array([1.0, 0.0, 5.0]) * t + 0.5 * np.array([0.0, 0.0, -g]) * t**2
    # The dynamics are quadratic in time, which RK4 integrates exactly.
    assert np.allclose(body.position, expected, atol=1e-10)


def test_torque_free_principal_axis_spin():
    omega = 3.0
    body = _make_body(
        cog=(0.0, 0.0, 0.0),
        initial_angular_velocity=_world_vector([0.0, 0.0, omega]),
    )
    solver = RK4()
    solver.register(body)
    dt, n_steps = 0.001, 1000
    for _ in range(n_steps):
        solver.step(dt)
    assert np.allclose(body.angular_velocity, [0.0, 0.0, omega], atol=1e-10)
    expected_R = _rodrigues_reference(np.array([0.0, 0.0, 1.0]), omega * dt * n_steps)
    assert np.allclose(body.rotation_matrix, expected_R, atol=1e-6)


def test_torque_free_tumble_conserves_momentum_and_energy():
    inertia = np.diag([1.0, 2.0, 3.0])
    omega0 = np.array([0.3, 1.0, 0.5])
    body = _make_body(
        inertia=inertia,
        cog=(0.0, 0.0, 0.0),
        initial_angular_velocity=_world_vector(omega0),
    )
    L0 = inertia @ omega0
    energy0 = 0.5 * omega0 @ L0

    solver = RK4()
    solver.register(body)
    for _ in range(2000):
        solver.step(0.001)

    R_nb = body.rotation_matrix
    inertia_n = R_nb @ inertia @ R_nb.T
    omega = body.angular_velocity
    L = inertia_n @ omega
    energy = 0.5 * omega @ L
    assert np.allclose(L, L0, atol=1e-6)
    assert np.isclose(energy, energy0, atol=1e-6)


def test_rk4_beats_euler_on_rotation():
    def integrate(solver_cls, dt: float) -> np.ndarray:
        body = _make_body(
            cog=(0.0, 0.0, 0.0),
            initial_angular_velocity=_world_vector([0.0, 0.0, 2.0]),
        )
        solver = solver_cls()
        solver.register(body)
        for _ in range(round(1.0 / dt)):
            solver.step(dt)
        return body.rotation_matrix

    expected = _rodrigues_reference(np.array([0.0, 0.0, 1.0]), 2.0)
    err_euler = np.abs(integrate(ForwardEuler, 0.01) - expected).max()
    err_rk4 = np.abs(integrate(RK4, 0.01) - expected).max()
    assert err_rk4 < 1e-3 * err_euler
