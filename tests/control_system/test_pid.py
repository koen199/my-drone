import numpy as np
import pytest

from simulations.control_system.pid import PID


def test_proportional_only():
    pid = PID(kp=2.0, size=3)
    np.testing.assert_allclose(pid.update([1.0, -0.5, 0.0], dt=0.01), [2.0, -1.0, 0.0])


def test_scalar_gain_is_broadcast_and_vector_gain_is_per_axis():
    np.testing.assert_allclose(PID(kp=3.0, size=3).update(np.ones(3), 0.01), [3.0, 3.0, 3.0])
    np.testing.assert_allclose(
        PID(kp=[1.0, 2.0, 3.0], size=3).update(np.ones(3), 0.01), [1.0, 2.0, 3.0]
    )


def test_integral_accumulates_in_output_units():
    """The integral term is Ki * sum(e * dt), not the bare sum(e * dt)."""
    pid = PID(kp=0.0, ki=2.0, size=1)
    for _ in range(10):
        output = pid.update([1.0], dt=0.1)
    assert output[0] == pytest.approx(2.0 * 1.0 * 1.0)
    assert pid.integral[0] == pytest.approx(2.0)


def test_derivative_is_zero_on_the_first_sample():
    """No previous error exists yet, so a step must not produce a spike."""
    pid = PID(kp=0.0, kd=1.0, size=1)
    assert pid.update([5.0], dt=0.1)[0] == pytest.approx(0.0)
    # Second call sees a constant error, so de/dt is still zero.
    assert pid.update([5.0], dt=0.1)[0] == pytest.approx(0.0)
    # A change of 1.0 over 0.1 s is a derivative of 10.
    assert pid.update([6.0], dt=0.1)[0] == pytest.approx(10.0)


def test_integral_limit_clamps_windup():
    pid = PID(kp=0.0, ki=10.0, size=1, integral_limit=1.5)
    for _ in range(100):
        output = pid.update([1.0], dt=0.1)
    assert output[0] == pytest.approx(1.5)


def test_output_limit_clamps_total():
    pid = PID(kp=100.0, size=2, output_limit=(-1.0, 2.0))
    np.testing.assert_allclose(pid.update([1.0, -1.0], 0.01), [2.0, -1.0])


def test_output_limit_accepts_a_one_sided_bound():
    pid = PID(kp=1.0, size=1, output_limit=(0.0, None))
    assert pid.update([-5.0], 0.01)[0] == pytest.approx(0.0)
    assert pid.update([1e6], 0.01)[0] == pytest.approx(1e6)


def test_initial_integral_preload():
    pid = PID(kp=0.0, ki=0.0, size=1, initial_integral=7.85)
    assert pid.update([0.0], 0.01)[0] == pytest.approx(7.85)


def test_reset_restores_the_preload():
    pid = PID(kp=0.0, ki=1.0, size=1, initial_integral=2.0)
    for _ in range(5):
        pid.update([1.0], 0.1)
    assert pid.integral[0] > 2.0
    pid.reset()
    assert pid.integral[0] == pytest.approx(2.0)


def test_reset_clears_the_derivative_memory():
    """Without a previous sample to difference against, de/dt is zero again."""
    pid = PID(kp=0.0, kd=1.0, size=1)
    pid.update([1.0], 0.1)
    assert pid.update([2.0], 0.1)[0] == pytest.approx(10.0)
    pid.reset()
    assert pid.update([100.0], 0.1)[0] == pytest.approx(0.0)


def test_reset_can_override_the_preload():
    pid = PID(kp=0.0, size=1, initial_integral=1.0)
    pid.reset(initial_integral=4.0)
    assert pid.integral[0] == pytest.approx(4.0)


def test_derivative_filter_lags_a_step():
    unfiltered = PID(kp=0.0, kd=1.0, size=1)
    filtered = PID(kp=0.0, kd=1.0, size=1, derivative_filter_tau=0.1)
    for pid in (unfiltered, filtered):
        pid.update([0.0], 0.01)
    raw = unfiltered.update([1.0], 0.01)[0]
    smooth = filtered.update([1.0], 0.01)[0]
    assert 0.0 < smooth < raw
    # tau/(tau+dt) of the step is deferred to later samples.
    assert smooth == pytest.approx(raw * 0.01 / 0.11)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        PID(kp=1.0, size=0)
    with pytest.raises(ValueError):
        PID(kp=[1.0, 2.0], size=3)
    with pytest.raises(ValueError):
        PID(kp=1.0, size=1, derivative_filter_tau=-1.0)
    with pytest.raises(ValueError):
        PID(kp=1.0, size=1).update([0.0], dt=0.0)
    with pytest.raises(ValueError):
        PID(kp=1.0, size=3).update([0.0, 1.0], dt=0.1)
