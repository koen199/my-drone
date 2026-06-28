"""AHRS simulation script.

The speed trajectory (body-frame angular velocity) is defined directly as a
function of time and integrated with the exact Rodrigues formula to produce
ground-truth orientations.  Sensor readings are derived from those orientations
and corrupted with configurable Gaussian noise.

Usage:
    python simulate.py
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from drone_control_system.ahrs import AHRSFilter, rotation_matrix_to_euler

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT = 0.01
DURATION_S = 30.0
N_STEPS = int(DURATION_S / DT)

GRAVITY_SPECIFIC_FORCE_NAV = np.array([0.0, 0.0, 9.81])  # specific force up, ENU hover
MAG_NORTH_NAV = np.array([0.0, 1.0, 0.0])

_FREQ = 0.2                       # Hz — sinusoidal sweep (phase 2)
_YAW_RATE = np.deg2rad(15.0)      # rad/s — steady yaw (phase 3)


@dataclass
class SensorNoiseConfig:
    gyro_std: float = np.deg2rad(0.01)   # rad/s
    accel_std: float = 0.05             # m/s²
    mag_std: float = 0.02               # normalised


# ---------------------------------------------------------------------------
# Speed trajectory
# ---------------------------------------------------------------------------

def omega_trajectory(t: float) -> np.ndarray:
    """Body-frame angular velocity [wx, wy, wz] at time t.

    0–10 s  : stationary hover
    10–20 s : sinusoidal roll/pitch sweep  (±20° / ±10° at 0.2 Hz)
    20–30 s : steady yaw rotation at 15 °/s
    """
    if t < 10.0:
        return np.zeros(3)
    elif t < 20.0:
        s = t - 10.0
        wx = np.deg2rad(20.0) * 2 * np.pi * _FREQ * np.cos(2 * np.pi * _FREQ * s)
        wy = np.deg2rad(10.0) * 2 * np.pi * _FREQ * np.cos(2 * np.pi * _FREQ * s + np.pi / 4)
        return np.array([wx, wy, 0.0])
    else:
        return np.array([0.0, 0.0, _YAW_RATE])


# ---------------------------------------------------------------------------
# Rodrigues integrator
# ---------------------------------------------------------------------------

def _rodrigues_exact(omega: np.ndarray, dt: float) -> np.ndarray:
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-12:
        return np.eye(3)
    axis = omega / np.linalg.norm(omega)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# Trajectory + sensor simulation
# ---------------------------------------------------------------------------

def run_trajectory_and_sensors(
    noise_cfg: SensorNoiseConfig | None = None,
    seed: int = 42,
):
    """Integrate the speed trajectory and produce ground-truth + noisy sensor data.

    Returns
    -------
    euler_true  : (N, 3) ground-truth roll/pitch/yaw in radians
    sensor_true : dict  'gyro', 'accel', 'mag' — noiseless (N, 3) arrays
    sensor_meas : dict  'gyro', 'accel', 'mag' — noisy (N, 3) arrays
    """
    if noise_cfg is None:
        noise_cfg = SensorNoiseConfig()
    rng = np.random.default_rng(seed)

    euler_true = np.zeros((N_STEPS, 3))
    gyro_true  = np.zeros((N_STEPS, 3))
    accel_true = np.zeros((N_STEPS, 3))
    mag_true   = np.zeros((N_STEPS, 3))
    gyro_meas  = np.zeros((N_STEPS, 3))
    accel_meas = np.zeros((N_STEPS, 3))
    mag_meas   = np.zeros((N_STEPS, 3))

    R = np.eye(3)
    for k in range(N_STEPS):
        t = k * DT
        omega = omega_trajectory(t)
        R = _rodrigues_exact(omega, DT) @ R

        gt = R @ GRAVITY_SPECIFIC_FORCE_NAV
        mt = R @ MAG_NORTH_NAV

        euler_true[k]  = rotation_matrix_to_euler(R)
        gyro_true[k]   = omega
        accel_true[k]  = gt
        mag_true[k]    = mt

        gyro_meas[k]   = omega + rng.normal(0.0, noise_cfg.gyro_std,  3) 
        accel_meas[k]  = gt    + rng.normal(0.0, noise_cfg.accel_std, 3)  
        mag_meas[k]    = mt    + rng.normal(0.0, noise_cfg.mag_std,   3) 

    sensor_true = {"gyro": gyro_true, "accel": accel_true, "mag": mag_true}
    sensor_meas = {"gyro": gyro_meas, "accel": accel_meas, "mag": mag_meas}
    return euler_true, sensor_true, sensor_meas


# ---------------------------------------------------------------------------
# AHRS filter loop
# ---------------------------------------------------------------------------

def run_ahrs(
    sensor_meas: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run AHRS filter over measured sensor data.

    Returns
    -------
    estimated     : (N, 3) AHRS Euler angles (rad)
    gravity_refs  : (N, 3) AHRS gravity EMA reference in body frame
    mag_refs      : (N, 3) AHRS mag EMA reference in body frame
    """
    gyro  = sensor_meas["gyro"]
    accel = sensor_meas["accel"]
    mag   = sensor_meas["mag"]
    n = len(gyro)

    ahrs = AHRSFilter(
        dt=DT,
        gravity_gain=0.01,
        mag_gain=0.01,
        ema_window_accel_s=60.0,
        ema_window_mag_s=60.0,
    )
    #ahrs.initialize(accel[0], mag[0])
    ahrs.initialize(GRAVITY_SPECIFIC_FORCE_NAV, MAG_NORTH_NAV)

    estimated    = np.zeros((n, 3))
    gravity_refs = np.zeros((n, 3))
    mag_refs     = np.zeros((n, 3))

    for k in range(n):
        roll, pitch, yaw = ahrs.update(gyro[k], accel[k], mag[k])
        estimated[k]    = [roll, pitch, yaw]
        gravity_refs[k] = ahrs.gravity_ref_body
        mag_refs[k]     = ahrs.mag_ref_body

    return estimated, gravity_refs, mag_refs


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def _print_summary(euler_true: np.ndarray, estimated: np.ndarray) -> None:
    warmup = int(5.0 / DT)
    labels = ["Roll", "Pitch", "Yaw"]
    print("\nAHRS simulation results (after 5 s warm-up):")
    print(f"{'Angle':<8}  {'RMS error (deg)':>16}")
    print("-" * 28)
    for i, label in enumerate(labels):
        err = np.rad2deg(_angle_diff(euler_true[warmup:, i], estimated[warmup:, i]))
        print(f"{label:<8}  {np.sqrt(np.mean(err ** 2)):>16.3f}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_euler_angles(euler_true: np.ndarray, estimated: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(len(euler_true)) * DT
    labels = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("AHRS Simulation — Ground Truth vs Estimate")
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, np.rad2deg(euler_true[:, i]), label="Ground truth", linewidth=1.5)
        ax.plot(t, np.rad2deg(estimated[:, i]), "--", label="AHRS estimate", linewidth=1.0)
        ax.set_ylabel(f"{label} (deg)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()


def _plot_sensor_comparison(sensor_true: dict, sensor_meas: dict) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(N_STEPS) * DT
    row_meta = [
        ("gyro",  "Gyro (rad/s)"),
        ("accel", "Accel (m/s²)"),
        ("mag",   "Mag (norm)"),
    ]
    axis_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    fig.suptitle("Sensor Data — Measured vs Actual (noise contribution)")

    for row, (key, row_label) in enumerate(row_meta):
        for col, ax_label in enumerate(axis_labels):
            ax = axes[row, col]
            ax.plot(t, sensor_true[key][:, col], label="Actual", linewidth=1.2)
            ax.plot(t, sensor_meas[key][:, col], alpha=0.5, linewidth=0.6, label="Measured")
            ax.set_ylabel(f"{row_label} {ax_label}")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"Axis {ax_label}")
            if row == 2:
                ax.set_xlabel("Time (s)")
    axes[0, 2].legend(fontsize=7, loc="upper right")
    plt.tight_layout()


def _plot_reference_vectors(
    sensor_true: dict,
    gravity_refs: np.ndarray,
    mag_refs: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(N_STEPS) * DT
    axis_labels = ["X", "Y", "Z"]
    rows = [
        ("Gravity ref (m/s²)", sensor_true["accel"], gravity_refs),
        ("Mag ref (norm)",     sensor_true["mag"],   mag_refs),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    fig.suptitle("Reference Vectors — True vs AHRS EMA Estimate")

    for row, (row_label, truth, ref) in enumerate(rows):
        for col, ax_label in enumerate(axis_labels):
            ax = axes[row, col]
            ax.plot(t, truth[:, col], label="True (noiseless)", linewidth=1.2)
            ax.plot(t, ref[:, col], "--", label="AHRS EMA ref", linewidth=1.0)
            ax.set_ylabel(f"{row_label} {ax_label}")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"Axis {ax_label}")
            if row == 1:
                ax.set_xlabel("Time (s)")
    axes[0, 2].legend(fontsize=7, loc="upper right")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_simulation(noise_cfg: SensorNoiseConfig | None = None) -> None:
    euler_true, sensor_true, sensor_meas = run_trajectory_and_sensors(noise_cfg)
    estimated, gravity_refs, mag_refs = run_ahrs(sensor_meas)

    _print_summary(euler_true, estimated)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    _plot_euler_angles(euler_true, estimated)
    _plot_sensor_comparison(sensor_true, sensor_meas)
    _plot_reference_vectors(sensor_true, gravity_refs, mag_refs)
    plt.show()


if __name__ == "__main__":
    run_simulation()
