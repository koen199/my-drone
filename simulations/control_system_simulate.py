"""Closed-loop control system simulation.

Flies the controller of docs/control_system/control_system.tex against the
rigid body simulator, which is the end-to-end check that the two fit together:

    setpoint -> ControlSystem -> Mixer -> Propellers -> RigidBody -> RK4
                     ^                                              |
                     '------------- state estimate -----------------'

The physics runs at 1 kHz and the controller at 250 Hz, so the loop sees a
slightly stale state exactly as it would on hardware. By default the state
estimate comes from the error-state Kalman filter of
docs/state_estimator/state_estimator.tex (simulations/state_estimator), fed
by the noisy, biased sensors of simulations/sensors.py:

    IMU 1 kHz (predict) | accel gravity 100 Hz | mag, baro 50 Hz | GPS 10 Hz

Setting ``Scenario.estimator = "perfect"`` flies on the true state instead.
Tracking plots always show the TRUE state; estimator.png shows the
estimation errors against the filter's own 3-sigma bounds.

Plots for every scenario are written as PNGs to output/control_system/<name>/.

Usage:
    python -m simulations.control_system_simulate
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from simulations.control_system import (
    ControlGains,
    ControlSystem,
    Propellers,
    QuadrotorGeometry,
    Setpoint,
    StateEstimate,
)
from simulations.math_helper import (
    quaternion_conjugate,
    quaternion_from_axis_angle,
    quaternion_multiply,
    rotation_matrix_to_euler,
)
from simulations.rigid_body_simulator import (
    RK4,
    CoordinateSystem,
    Force,
    RigidBody,
    Vector,
)
from simulations.sensors import SensorNoise, SensorSuite
from simulations.state_estimator import EskfNoise, ErrorStateKalmanFilter

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT = 0.001                 # physics step and IMU rate, 1 kHz
CONTROL_DECIMATION = 4     # controller runs at 250 Hz
ACCEL_UPDATE_DECIMATION = 10   # accelerometer gravity update, 100 Hz
MAG_DECIMATION = 20            # magnetometer, 50 Hz
BARO_DECIMATION = 20           # barometer, 50 Hz
GPS_DECIMATION = 100           # GNSS, 10 Hz
GRAVITY = 9.81

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "control_system"

# ---------------------------------------------------------------------------
# Vehicle: a ~1.2 kg quadrotor. The arms are deliberately unequal (a
# rectangular rather than square frame) so roll and pitch have different
# authority, which is a better test of the mixer than a symmetric frame.
# ---------------------------------------------------------------------------
MASS = 1.2
INERTIA = np.diag([0.011, 0.011, 0.021])
COG = np.array([0.0, 0.0, 0.0])   # CoG at the body origin

GEOMETRY = QuadrotorGeometry(
    roll_arm=0.17,
    pitch_arm=0.13,
    thrust_coefficient=6.0e-6,      # ~2.9 N per motor at 700 rad/s
    drag_coefficient=9.6e-8,        # c = k_d/k_t = 0.016 m
    min_motor_speed=0.0,
    max_motor_speed=1200.0,         # thrust-to-weight ~2.9
)


@dataclass
class Scenario:
    """A named flight case.

    ``setpoint(t)`` returns the commanded altitude and attitude at time t;
    ``initial_orientation`` optionally starts the drone away from level.
    ``estimator`` is "eskf" (the Kalman filter on simulated sensors) or
    "perfect" (read the true state off the rigid body).
    """

    name: str
    title: str
    setpoint: Callable[[float], Setpoint]
    duration: float
    initial_orientation: np.ndarray | None = None
    gains: ControlGains = field(default_factory=ControlGains)
    estimator: str = "eskf"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def _takeoff_and_hover(t: float) -> Setpoint:
    """Climb to 5 m, hold, then descend to 2 m."""
    if t < 1.0:
        return Setpoint(altitude=0.0)
    if t < 12.0:
        return Setpoint(altitude=5.0)
    return Setpoint(altitude=2.0)


def _attitude_steps(t: float) -> Setpoint:
    """Hold 3 m while stepping roll, pitch and yaw in turn."""
    setpoint = Setpoint(altitude=3.0)
    if 3.0 <= t < 6.0:
        setpoint.roll = np.deg2rad(20.0)
    elif 6.0 <= t < 9.0:
        setpoint.pitch = np.deg2rad(-20.0)
    elif 9.0 <= t < 12.0:
        setpoint.yaw = np.deg2rad(45.0)
    elif t >= 12.0:
        setpoint.roll = np.deg2rad(15.0)
        setpoint.pitch = np.deg2rad(15.0)
        setpoint.yaw = np.deg2rad(45.0)
    return setpoint


def _upset_recovery(t: float) -> Setpoint:
    """Level off and hold altitude after being released in a steep attitude."""
    return Setpoint(altitude=0.0)


def _upset_orientation() -> np.ndarray:
    """Quaternion for 50 deg roll then -35 deg pitch, scalar-first."""
    return quaternion_multiply(
        quaternion_from_axis_angle([0.0, 1.0, 0.0], np.deg2rad(-35.0)),
        quaternion_from_axis_angle([1.0, 0.0, 0.0], np.deg2rad(50.0)),
    )


SCENARIOS = [
    Scenario(
        name="takeoff_and_hover",
        title="Altitude Tracking - Climb, Hover, Descend",
        setpoint=_takeoff_and_hover,
        duration=20.0,
    ),
    Scenario(
        name="attitude_steps",
        title="Attitude Tracking - Roll / Pitch / Yaw Steps at Constant Altitude",
        setpoint=_attitude_steps,
        duration=16.0,
    ),
    Scenario(
        name="upset_recovery",
        title="Upset Recovery - Released at 50 deg Roll / -35 deg Pitch",
        setpoint=_upset_recovery,
        duration=10.0,
        initial_orientation=_upset_orientation(),
    ),
]


# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------

def measure(drone: RigidBody) -> StateEstimate:
    """Perfect state estimator reading the rigid body's true state.

    Note the body rates come from get_rotation_speed() with a BODY-frame
    probe point, which is what a strapdown gyro would report; the rigid body
    stores omega in the world frame internally.
    """
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


def estimate_state(eskf: ErrorStateKalmanFilter, gyro: np.ndarray) -> StateEstimate:
    """Controller input from the Kalman filter's nominal state.

    The body rates are the latest raw gyro sample minus the estimated bias:
    the filter does not smooth them, just as a flight controller closes its
    rate loop on the (bias-corrected) gyro directly.
    """
    state = eskf.state
    roll, pitch, yaw = eskf.euler
    return StateEstimate(
        altitude=state.p[2],
        vertical_speed=state.v[2],
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        body_rates=eskf.body_rates(gyro),
    )


def attitude_error(q_estimate: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Small-angle body-frame error dtheta with q_true = q_estimate (x) dq(dtheta).

    This is the filter's own attitude error-state definition, so it can be
    compared directly against the dtheta 3-sigma bounds.
    """
    dq = quaternion_multiply(quaternion_conjugate(q_estimate), q_true)
    if dq[0] < 0.0:
        dq = -dq
    return 2.0 * dq[1:]


def _build_estimator(
    drone: RigidBody, sensors: SensorSuite
) -> tuple[ErrorStateKalmanFilter, np.ndarray, np.ndarray]:
    """Initialise the ESKF from the first IMU / magnetometer sample.

    At t = 0 no forces are applied yet, so the accelerometer reads a pure
    gravity reaction and the TRIAD attitude is exact up to sensor noise.
    Position and velocity start from the (known) launch point.
    """
    n = sensors.noise
    eskf = ErrorStateKalmanFilter(EskfNoise(
        sigma_a=n.sigma_a,
        sigma_g=n.sigma_g,
        sigma_gps_pos=n.sigma_gps_pos,
        sigma_gps_vel=n.sigma_gps_vel,
        sigma_baro=n.sigma_baro,
        sigma_mag=n.sigma_mag,
        gravity=n.gravity,
        mag_reference=n.mag_reference,
    ))
    gyro, accel = sensors.imu()
    eskf.initialize(drone.position, drone.velocity, accel, sensors.mag())
    return eskf, gyro, accel


def run_closed_loop(scenario: Scenario) -> dict[str, np.ndarray]:
    """Fly one scenario and return the logged timeseries (per control step)."""
    if scenario.estimator not in ("eskf", "perfect"):
        raise ValueError(f"unknown estimator {scenario.estimator!r}")
    use_eskf = scenario.estimator == "eskf"

    cog_vector = Vector(components=COG, coordinate_system=CoordinateSystem.BODY)
    drone = RigidBody(
        inertia_body=INERTIA,
        mass=MASS,
        cog_position=cog_vector,
        initial_orientation=scenario.initial_orientation,
    )
    propellers = Propellers(GEOMETRY, cog_position=cog_vector)
    control_system = ControlSystem(MASS, INERTIA, GEOMETRY, gains=scenario.gains)

    sensors = SensorSuite(drone, cog_vector, imu_dt=DT, noise=SensorNoise())
    if use_eskf:
        eskf, gyro, accel = _build_estimator(drone, sensors)

    # Gravity is a world-frame force: it keeps pointing down as the drone
    # rolls, so it must NOT be expressed in the body frame.
    gravity = Force(
        application_point=cog_vector,
        force=Vector(
            components=[0.0, 0.0, -MASS * GRAVITY],
            coordinate_system=CoordinateSystem.WORLD,
        ),
    )

    solver = RK4()
    solver.register(drone)

    log: dict[str, list] = {
        key: []
        for key in (
            "time", "altitude", "altitude_sp", "vertical_speed", "vertical_speed_sp",
            "attitude", "attitude_sp", "body_rates", "body_rates_sp",
            "torque", "vertical_force", "motor_speeds", "motor_thrusts", "saturated",
            # Estimator diagnostics (truth vs estimate).
            "altitude_est", "attitude_est", "position", "position_est",
            "velocity", "velocity_est", "attitude_error", "gyro_bias_est",
            "accel_bias_est", "sigma",
        )
    }
    control_dt = DT * CONTROL_DECIMATION

    for step in range(int(scenario.duration / DT)):
        t = solver.time

        if use_eskf:
            # Measurement updates at time t. The accelerometer update reuses
            # the IMU sample that drove the last prediction.
            if step % ACCEL_UPDATE_DECIMATION == 0:
                eskf.update_accel(accel)
            if step % MAG_DECIMATION == 0:
                eskf.update_mag(sensors.mag())
            if step % BARO_DECIMATION == 0:
                eskf.update_baro(sensors.baro())
            if step % GPS_DECIMATION == 0:
                eskf.update_gps(*sensors.gps())

        if step % CONTROL_DECIMATION == 0:
            setpoint = scenario.setpoint(t)
            estimate = estimate_state(eskf, gyro) if use_eskf else measure(drone)
            output = control_system.update(setpoint, estimate, control_dt)

            loads = propellers.set_speed(output.motor_speeds)
            drone.apply_forces(loads.forces + [gravity])
            drone.apply_torques(loads.torques)

            roll, pitch, yaw = rotation_matrix_to_euler(drone.rotation_matrix)
            log["time"].append(t)
            log["altitude"].append(drone.position[2])
            log["altitude_sp"].append(setpoint.altitude)
            log["vertical_speed"].append(drone.velocity[2])
            log["vertical_speed_sp"].append(output.vertical_speed_setpoint)
            log["attitude"].append([roll, pitch, yaw])
            log["attitude_sp"].append(list(setpoint.attitude))
            log["body_rates"].append(estimate.body_rates)
            log["body_rates_sp"].append(output.body_rate_setpoint)
            log["torque"].append(output.torque)
            log["vertical_force"].append(output.vertical_force)
            log["motor_speeds"].append(output.motor_speeds)
            log["motor_thrusts"].append(output.motor_thrusts)
            log["saturated"].append(output.saturated)

            log["altitude_est"].append(estimate.altitude)
            log["attitude_est"].append(list(estimate.attitude))
            log["position"].append(drone.position)
            log["velocity"].append(drone.velocity)
            if use_eskf:
                state = eskf.state
                log["position_est"].append(state.p)
                log["velocity_est"].append(state.v)
                log["attitude_error"].append(attitude_error(state.q, drone.orientation))
                log["gyro_bias_est"].append(state.b_g)
                log["accel_bias_est"].append(state.b_a)
                log["sigma"].append(eskf.sigma)

        if use_eskf:
            # Sample the IMU with this step's forces applied and hold it over
            # the step, exactly as the physics holds the forces.
            gyro, accel = sensors.imu()
            solver.step(DT)
            eskf.predict(gyro, accel, DT)
        else:
            solver.step(DT)

    result = {key: np.asarray(value) for key, value in log.items()}
    result["gyro_bias"] = sensors.noise.gyro_bias
    result["accel_bias"] = sensors.noise.accel_bias
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(scenario: Scenario, log: dict[str, np.ndarray]) -> None:
    """Report steady-state tracking error over the last second of flight."""
    tail = log["time"] >= log["time"][-1] - 1.0
    labels = ["Altitude (m)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
    errors = [
        log["altitude"][tail] - log["altitude_sp"][tail],
        *[
            np.rad2deg(log["attitude"][tail, i] - log["attitude_sp"][tail, i])
            for i in range(3)
        ],
    ]

    print(f"\n{scenario.title}")
    print(f"{'Channel':<16}{'mean error':>13}{'max |error|':>14}")
    print("-" * 43)
    for label, error in zip(labels, errors):
        print(f"{label:<16}{np.mean(error):>13.4f}{np.max(np.abs(error)):>14.4f}")
    saturated = float(np.mean(log["saturated"])) * 100.0
    print(f"{'Motor saturation':<16}{saturated:>12.1f} % of steps")

    if log["position_est"].size:
        _print_estimator_summary(log)


def _print_estimator_summary(log: dict[str, np.ndarray]) -> None:
    """RMS estimation error over the whole flight, per axis."""
    def rms(error: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(error**2, axis=0))

    rows = [
        ("Position (m)", rms(log["position_est"] - log["position"])),
        ("Velocity (m/s)", rms(log["velocity_est"] - log["velocity"])),
        ("Attitude (deg)", np.rad2deg(rms(log["attitude_error"]))),
        ("Gyro bias (deg/s)", np.rad2deg(log["gyro_bias_est"][-1] - log["gyro_bias"])),
        ("Accel bias (m/s2)", log["accel_bias_est"][-1] - log["accel_bias"]),
    ]
    print(f"\n{'Estimator':<20}{'x':>10}{'y':>10}{'z':>10}   (RMS error; bias: final error)")
    print("-" * 50)
    for label, values in rows:
        print(f"{label:<20}" + "".join(f"{v:>10.4f}" for v in values))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_altitude(log: dict[str, np.ndarray], title: str):
    t = log["time"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{title} — Vertical Channel")

    axes[0].plot(t, log["altitude_sp"], "--", label="Setpoint", linewidth=1.2)
    axes[0].plot(t, log["altitude"], label="Actual", linewidth=1.4)
    axes[0].set_ylabel("Altitude (m)")

    axes[1].plot(t, log["vertical_speed_sp"], "--", label="Setpoint (saturated)", linewidth=1.2)
    axes[1].plot(t, log["vertical_speed"], label="Actual", linewidth=1.4)
    axes[1].set_ylabel("Vertical speed (m/s)")
    axes[1].set_xlabel("Time (s)")

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_attitude(log: dict[str, np.ndarray], title: str):
    t = log["time"]
    labels = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{title} — Attitude")
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, np.rad2deg(log["attitude_sp"][:, i]), "--", label="Setpoint", linewidth=1.2)
        ax.plot(t, np.rad2deg(log["attitude"][:, i]), label="Actual", linewidth=1.4)
        ax.set_ylabel(f"{label} (deg)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


def _plot_rate_loop(log: dict[str, np.ndarray], title: str):
    """Inner loop: how well the measured body rates track their setpoints."""
    t = log["time"]
    labels = ["p (roll rate)", "q (pitch rate)", "r (yaw rate)"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{title} — Body Rate Loop")
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, log["body_rates_sp"][:, i], "--", label="Setpoint", linewidth=1.2)
        ax.plot(t, log["body_rates"][:, i], label="Measured", linewidth=1.2)
        ax.set_ylabel(f"{label} (rad/s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


def _plot_actuators(log: dict[str, np.ndarray], title: str):
    """Mixer output: the control effort and the motor commands it becomes."""
    t = log["time"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"{title} — Control Effort and Actuators")

    axes[0].plot(t, log["vertical_force"], label="$F_z$", linewidth=1.2)
    axes[0].axhline(MASS * GRAVITY, color="black", linewidth=0.6, alpha=0.5, label="Hover $mg$")
    axes[0].set_ylabel("Vertical force (N)")

    for i, label in enumerate(["$T_x$", "$T_y$", "$T_z$"]):
        axes[1].plot(t, log["torque"][:, i], label=label, linewidth=1.0)
    axes[1].set_ylabel("Torque (N·m)")

    for i in range(4):
        axes[2].plot(t, log["motor_speeds"][:, i], label=f"M{i + 1}", linewidth=1.0)
    axes[2].axhline(
        GEOMETRY.max_motor_speed, color="red", linewidth=0.6, alpha=0.5, label="Limit"
    )
    axes[2].set_ylabel("Motor speed (rad/s)")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_estimator(log: dict[str, np.ndarray], title: str):
    """ESKF errors against the filter's own 3-sigma bounds, and bias estimates.

    Columns are the x / y / z axes; rows are position, velocity and attitude
    error (estimate vs truth, shaded +/-3 sigma), then the gyro and
    accelerometer bias estimates against the true sensor biases.
    """
    t = log["time"]
    sigma = log["sigma"]
    axis_names = ["x", "y", "z"]
    rows = [
        ("Position err (m)", log["position_est"] - log["position"], sigma[:, 0:3], 1.0),
        ("Velocity err (m/s)", log["velocity_est"] - log["velocity"], sigma[:, 3:6], 1.0),
        # attitude_error is truth relative to estimate; negate for est - truth.
        ("Attitude err (deg)", -log["attitude_error"], sigma[:, 6:9], np.rad2deg(1.0)),
    ]
    fig, axes = plt.subplots(5, 3, figsize=(14, 13), sharex=True)
    fig.suptitle(f"{title} — State Estimator")

    for row, (label, error, bound, scale) in enumerate(rows):
        for i in range(3):
            ax = axes[row, i]
            ax.fill_between(
                t, -3.0 * bound[:, i] * scale, 3.0 * bound[:, i] * scale,
                color="tab:gray", alpha=0.25, label="±3σ",
            )
            ax.plot(t, error[:, i] * scale, linewidth=0.9, label="Error")
            ax.set_title(f"{label.split(' ')[0]} {axis_names[i]}", fontsize=9)
            if i == 0:
                ax.set_ylabel(label)

    bias_rows = [
        ("Gyro bias (deg/s)", log["gyro_bias_est"], log["gyro_bias"], sigma[:, 9:12], np.rad2deg(1.0)),
        ("Accel bias (m/s²)", log["accel_bias_est"], log["accel_bias"], sigma[:, 12:15], 1.0),
    ]
    for offset, (label, estimate, truth, bound, scale) in enumerate(bias_rows):
        row = len(rows) + offset
        for i in range(3):
            ax = axes[row, i]
            est = estimate[:, i] * scale
            ax.fill_between(
                t, est - 3.0 * bound[:, i] * scale, est + 3.0 * bound[:, i] * scale,
                color="tab:gray", alpha=0.25, label="±3σ",
            )
            ax.plot(t, est, linewidth=1.0, label="Estimate")
            ax.axhline(truth[i] * scale, color="black", linestyle="--", linewidth=0.8, label="True")
            ax.set_title(f"{label.split(' ')[0]} bias {axis_names[i]}", fontsize=9)
            if i == 0:
                ax.set_ylabel(label)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_scenario(scenario: Scenario) -> dict[str, np.ndarray]:
    """Run one scenario end-to-end and save its plots to output/<name>/."""
    print(f"=== Scenario: {scenario.name} ===")
    log = run_closed_loop(scenario)
    _print_summary(scenario, log)

    figures = {
        "altitude": _plot_altitude(log, scenario.title),
        "attitude": _plot_attitude(log, scenario.title),
        "rate_loop": _plot_rate_loop(log, scenario.title),
        "actuators": _plot_actuators(log, scenario.title),
    }
    if log["position_est"].size:
        figures["estimator"] = _plot_estimator(log, scenario.title)

    out_dir = OUTPUT_DIR / scenario.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    plt.close("all")
    return log


def run_simulation(scenarios: list[Scenario] | None = None) -> None:
    for scenario in scenarios or SCENARIOS:
        run_scenario(scenario)


if __name__ == "__main__":
    run_simulation()
