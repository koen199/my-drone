import numpy as np
import pytest

from simulations.control_system.altitude import AltitudeController


def test_proportional_law():
    controller = AltitudeController(kp=1.5, min_climb_rate=-10.0, max_climb_rate=10.0)
    assert controller.update(altitude_setpoint=2.0, altitude=0.0) == pytest.approx(3.0)


def test_zero_error_commands_zero_speed():
    controller = AltitudeController(kp=2.0)
    assert controller.update(5.0, 5.0) == pytest.approx(0.0)


def test_below_setpoint_commands_a_climb():
    controller = AltitudeController(kp=1.0)
    assert controller.update(altitude_setpoint=3.0, altitude=1.0) > 0.0


def test_above_setpoint_commands_a_sink():
    controller = AltitudeController(kp=1.0)
    assert controller.update(altitude_setpoint=1.0, altitude=3.0) < 0.0


def test_climb_and_sink_saturation():
    controller = AltitudeController(kp=5.0, min_climb_rate=-1.0, max_climb_rate=2.0)
    assert controller.update(100.0, 0.0) == pytest.approx(2.0)
    assert controller.update(-100.0, 0.0) == pytest.approx(-1.0)


def test_saturation_limits_may_be_asymmetric():
    """Sinking is usually limited harder than climbing, so the two differ."""
    controller = AltitudeController(kp=10.0, min_climb_rate=-0.5, max_climb_rate=4.0)
    assert controller.climb_rate_limits == (-0.5, 4.0)


def test_is_stateless():
    """Repeated calls with the same inputs must give the same output."""
    controller = AltitudeController(kp=1.0)
    outputs = [controller.update(1.0, 0.0) for _ in range(5)]
    assert np.allclose(outputs, outputs[0])


def test_invalid_arguments():
    with pytest.raises(ValueError):
        AltitudeController(kp=-1.0)
    with pytest.raises(ValueError):
        AltitudeController(kp=1.0, min_climb_rate=0.5)
    with pytest.raises(ValueError):
        AltitudeController(kp=1.0, max_climb_rate=-0.5)
