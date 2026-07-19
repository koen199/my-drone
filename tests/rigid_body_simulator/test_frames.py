import numpy as np
import pytest

from simulations.rigid_body_simulator.frames import CoordinateSystem, Force, Torque, Vector


def test_vector_coerces_list_to_float_array():
    v = Vector(components=[1, 2, 3], coordinate_system=CoordinateSystem.BODY)
    assert isinstance(v.components, np.ndarray)
    assert v.components.dtype == float
    assert np.allclose(v.components, [1.0, 2.0, 3.0], atol=1e-12)


def test_vector_copies_input_array():
    raw = np.array([1.0, 2.0, 3.0])
    v = Vector(components=raw, coordinate_system=CoordinateSystem.WORLD)
    raw[0] = 99.0
    assert v.components[0] == 1.0


def test_vector_bad_shape_raises():
    with pytest.raises(ValueError):
        Vector(components=[1.0, 2.0], coordinate_system=CoordinateSystem.BODY)
    with pytest.raises(ValueError):
        Vector(components=np.eye(3), coordinate_system=CoordinateSystem.BODY)


def test_vector_accepts_coordinate_system_string():
    v = Vector(components=[0.0, 0.0, 1.0], coordinate_system="BODY")
    assert v.coordinate_system is CoordinateSystem.BODY


def test_vector_invalid_coordinate_system_raises():
    with pytest.raises(ValueError):
        Vector(components=[0.0, 0.0, 1.0], coordinate_system="GRAVITY")


def test_force_and_torque_hold_vectors():
    point = Vector(components=[0.0, 0.33, 0.1], coordinate_system=CoordinateSystem.BODY)
    force = Vector(components=[0.0, 0.0, -9.81], coordinate_system=CoordinateSystem.WORLD)
    f = Force(application_point=point, force=force)
    assert f.application_point.coordinate_system is CoordinateSystem.BODY
    assert f.force.coordinate_system is CoordinateSystem.WORLD

    tau = Torque(torque=Vector(components=[0.1, 0.0, 0.0], coordinate_system=CoordinateSystem.BODY))
    assert np.allclose(tau.torque.components, [0.1, 0.0, 0.0], atol=1e-12)
