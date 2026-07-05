"""AHRS simulation script.

Each scenario prescribes two things as functions of time:

* the body-frame **angular velocity** ``omega(t)``, integrated with the exact
  Rodrigues formula to give the ground-truth attitude, and
* the nav-frame **linear acceleration** ``a(t)`` (optional; zero for pure
  rotations).

The accelerometer then measures the true specific force ``f = a - g`` rotated
into the body frame.  When a manoeuvre carries linear acceleration, the
accelerometer no longer points along true up — precisely what fools a static
gravity-based attitude reference.

Each :class:`Scenario` pairs a name with a callable that produces the actual and
sensor timeseries.  Plots for every scenario are written as PNGs to
``output/<scenario-name>/``.

Usage:
    python simulate.py
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from drone_control_system.ahrs import AHRSFilter, rotation_matrix_to_euler

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT = 0.01

# Navigation frame is ENU: x = East, y = North, z = Up.
GRAVITY_NAV = np.array([0.0, 0.0, -9.81])            # gravitational acceleration
MAG_NORTH_NAV = np.array([0.0, 1.0, 0.0])

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ---- rotation-sweep scenario --------------------------------------------------
SWEEP_DURATION_S = 30.0
_FREQ = 0.2                       # Hz — sinusoidal sweep (phase 2)
_YAW_RATE = np.deg2rad(15.0)      # rad/s — steady yaw (phase 3)


@dataclass
class SensorNoiseConfig:
    gyro_std: float = np.deg2rad(0.01)   # rad/s
    accel_std: float = 0.05             # m/s²
    mag_std: float = 0.02               # normalised


@dataclass
class Scenario:
    """A named simulation case.

    ``generate(noise_cfg, seed)`` returns ``(euler_true, sensor_true,
    sensor_meas)``:
        euler_true  : (N, 3) ground-truth roll/pitch/yaw in radians
        sensor_true : dict 'gyro'/'accel'/'mag' — noiseless (N, 3) arrays
        sensor_meas : dict 'gyro'/'accel'/'mag' — noisy (N, 3) arrays
    """
    name: str
    generate: Callable[[SensorNoiseConfig | None, int], tuple[np.ndarray, dict, dict]]


# ---------------------------------------------------------------------------
# Rotation helpers
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
# Sensor generation
# ---------------------------------------------------------------------------

def _add_noise(sensor_true: dict, noise_cfg: SensorNoiseConfig, rng) -> dict:
    n = len(sensor_true["gyro"])
    return {
        "gyro":  sensor_true["gyro"]  + rng.normal(0.0, noise_cfg.gyro_std,  (n, 3)),
        "accel": sensor_true["accel"] + rng.normal(0.0, noise_cfg.accel_std, (n, 3)),
        "mag":   sensor_true["mag"]   + rng.normal(0.0, noise_cfg.mag_std,   (n, 3)),
    }


def _simulate(
    omega_fn: Callable[[float], np.ndarray],
    accel_nav_fn: Callable[[float], np.ndarray] | None,
    duration: float,
    noise_cfg: SensorNoiseConfig | None,
    seed: int,
):
    """Generate a scenario from a prescribed angular velocity and acceleration.

    ``omega_fn`` gives the body-frame angular velocity (integrated for the
    ground-truth attitude); ``accel_nav_fn`` gives the nav-frame linear
    acceleration (zero if omitted).  The accelerometer measures the true
    specific force ``f = a - g`` rotated into the body frame, so it captures the
    manoeuvre's linear acceleration on top of gravity.
    """
    if noise_cfg is None:
        noise_cfg = SensorNoiseConfig()
    rng = np.random.default_rng(seed)
    n = int(duration / DT)

    euler_true = np.zeros((n, 3))
    gyro_true  = np.zeros((n, 3))
    accel_true = np.zeros((n, 3))
    mag_true   = np.zeros((n, 3))

    R = np.eye(3)
    for k in range(n):
        t = k * DT
        omega = omega_fn(t)
        R = _rodrigues_exact(omega, DT) @ R

        a_nav = np.zeros(3) if accel_nav_fn is None else np.asarray(accel_nav_fn(t), dtype=float)
        f_nav = a_nav - GRAVITY_NAV

        euler_true[k] = rotation_matrix_to_euler(R)
        gyro_true[k]  = omega
        accel_true[k] = R @ f_nav
        mag_true[k]   = R @ MAG_NORTH_NAV

    sensor_true = {"gyro": gyro_true, "accel": accel_true, "mag": mag_true}
    return euler_true, sensor_true, _add_noise(sensor_true, noise_cfg, rng)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def omega_trajectory(t: float) -> np.ndarray:
    """Body-frame angular velocity [wx, wy, wz] for the rotation-sweep scenario.

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


def _sweep_generate(noise_cfg: SensorNoiseConfig | None = None, seed: int = 42):
    return _simulate(omega_trajectory, None, SWEEP_DURATION_S, noise_cfg, seed)


SCENARIOS = [
    Scenario("rotation_sweep", _sweep_generate),
]


# ---------------------------------------------------------------------------
# AHRS filter loop
# ---------------------------------------------------------------------------

def run_ahrs(sensor_meas: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run AHRS filter over measured sensor data.

    Returns
    -------
    estimated  : (N, 3) fused AHRS Euler angles (rad)
    triad      : (N, 3) absolute TRIAD orientation estimate Euler angles (rad)
    """
    gyro  = sensor_meas["gyro"]
    accel = sensor_meas["accel"]
    mag   = sensor_meas["mag"]
    n = len(gyro)

    ahrs = AHRSFilter(dt=DT, correction_gain=0.01)
    ahrs.initialize(accel[0], mag[0])

    estimated = np.zeros((n, 3))
    triad     = np.zeros((n, 3))

    for k in range(n):
        roll, pitch, yaw = ahrs.update(gyro[k], accel[k], mag[k])
        estimated[k] = [roll, pitch, yaw]
        R_triad = ahrs.triad_matrix
        triad[k] = rotation_matrix_to_euler(R_triad) if R_triad is not None else np.nan

    return estimated, triad


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

def _plot_euler_angles(euler_true: np.ndarray, estimated: np.ndarray):
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
    return fig


def _plot_estimate_error(euler_true: np.ndarray, estimated: np.ndarray):
    """Plot the error between ground truth and the fused AHRS estimate."""
    t = np.arange(len(euler_true)) * DT
    labels = ["Roll", "Pitch", "Yaw"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("AHRS Estimate Error (Ground Truth − Estimate)")
    for i, (ax, label) in enumerate(zip(axes, labels)):
        err = np.rad2deg(_angle_diff(euler_true[:, i], estimated[:, i]))
        ax.plot(t, err, label="Error", linewidth=1.0, color="tab:red")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(f"{label} error (deg)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


def _plot_sensor_comparison(sensor_true: dict, sensor_meas: dict):
    t = np.arange(len(sensor_true["gyro"])) * DT
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
    return fig


def _plot_triad_vs_fused(
    euler_true: np.ndarray,
    estimated: np.ndarray,
    triad: np.ndarray,
):
    """Compare the raw (noisy) TRIAD absolute estimate against the fused estimate.

    The TRIAD estimate responds instantaneously but is noisy; the fused estimate
    blends it into the gyro integration through the correction gain, yielding a
    smooth track that still follows the absolute reference over time.
    """
    t = np.arange(len(euler_true)) * DT
    labels = ["Roll", "Pitch", "Yaw"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Absolute TRIAD Estimate vs Fused AHRS Estimate")

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, np.rad2deg(triad[:, i]), label="TRIAD (raw)",
                linewidth=0.6, alpha=0.5)
        ax.plot(t, np.rad2deg(estimated[:, i]), label="Fused AHRS", linewidth=1.2)
        ax.plot(t, np.rad2deg(euler_true[:, i]), "--", label="True", linewidth=1.0)
        ax.set_ylabel(f"{label} (deg)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: Scenario,
    noise_cfg: SensorNoiseConfig | None = None,
    seed: int = 42,
) -> None:
    """Run one scenario end-to-end and save its plots to output/<name>/."""
    print(f"=== Scenario: {scenario.name} ===")
    euler_true, sensor_true, sensor_meas = scenario.generate(noise_cfg, seed)
    estimated, triad = run_ahrs(sensor_meas)

    _print_summary(euler_true, estimated)

    figures = {
        "euler_angles": _plot_euler_angles(euler_true, estimated),
        "estimate_error": _plot_estimate_error(euler_true, estimated),
        "sensor_comparison": _plot_sensor_comparison(sensor_true, sensor_meas),
        "triad_vs_fused": _plot_triad_vs_fused(euler_true, estimated, triad),
    }

    out_dir = OUTPUT_DIR / scenario.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    plt.close("all")


def run_simulation(
    scenarios: list[Scenario] | None = None,
    noise_cfg: SensorNoiseConfig | None = None,
) -> None:
    for scenario in scenarios or SCENARIOS:
        run_scenario(scenario, noise_cfg)


if __name__ == "__main__":
    run_simulation()
