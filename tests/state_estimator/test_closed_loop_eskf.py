"""
Closed-loop test: the controller flying on the ESKF estimate from noisy,
biased simulated sensors, rather than on the true state.
"""
import numpy as np

from simulations.control_system import Setpoint
from simulations.control_system_simulate import Scenario, run_closed_loop


def test_hover_on_eskf_estimate_reaches_altitude():
    scenario = Scenario(
        name="eskf_hover",
        title="ESKF hover",
        setpoint=lambda t: Setpoint(altitude=2.0),
        duration=8.0,
    )
    log = run_closed_loop(scenario)

    tail = log["time"] >= log["time"][-1] - 0.5
    # True altitude (not the estimate) settles on the setpoint.
    assert np.max(np.abs(log["altitude"][tail] - 2.0)) < 0.1
    # The vehicle stays near level and the estimator tracks the truth.
    assert np.max(np.abs(np.rad2deg(log["attitude"][tail, :2]))) < 3.0
    assert np.max(np.abs(log["position_est"] - log["position"])) < 0.5
    assert np.max(np.rad2deg(np.abs(log["attitude_error"]))) < 5.0
