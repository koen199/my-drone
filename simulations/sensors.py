"""
Simulated sensor suite for the state estimator.

Reads the rigid body's true state and produces the raw measurements the ESKF
of simulations/state_estimator consumes: a strapdown IMU (gyro + specific
force, with constant biases and white noise), GNSS position/velocity, a
barometer and a magnetometer. Keeping this outside the estimator means the
filter never sees the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulations.rigid_body_simulator import CoordinateSystem, RigidBody, Vector


@dataclass
class SensorNoise:
    """True sensor characteristics (the filter's tuning may differ).

    IMU noise is given as continuous densities; a sample taken at rate
    1/dt has standard deviation sigma / sqrt(dt).
    """

    sigma_a: float = 2.0e-3          # m/s^2/sqrt(Hz)
    sigma_g: float = 3.0e-4          # rad/s/sqrt(Hz)
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.01, -0.008, 0.005]))
    accel_bias: np.ndarray = field(default_factory=lambda: np.array([0.08, -0.05, 0.1]))
    sigma_gps_pos: float = 0.5       # m
    sigma_gps_vel: float = 0.05      # m/s
    sigma_baro: float = 0.1          # m
    sigma_mag: float = 0.005         # Gauss
    gravity: float = 9.81
    mag_reference: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.22, -0.42])  # Gauss, ENU, north + down
    )


class SensorSuite:
    """Noisy sensors mounted at the vehicle's centre of gravity.

    Parameters
    ----------
    drone : RigidBody
        The body being measured.
    cog_position : Vector
        CoG in the BODY frame; the IMU sits here so it sees no lever-arm
        acceleration.
    imu_dt : float
        IMU sample period, sets the per-sample IMU noise.
    noise : SensorNoise, optional
    rng : numpy Generator, optional
        Defaults to a fixed seed so runs are reproducible.
    """

    def __init__(
        self,
        drone: RigidBody,
        cog_position: Vector,
        imu_dt: float,
        noise: SensorNoise | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.drone = drone
        self.noise = noise or SensorNoise()
        self.rng = rng or np.random.default_rng(0)
        self._probe = cog_position
        self._imu_dt = imu_dt
        self._g_n = np.array([0.0, 0.0, -self.noise.gravity])

    def _white(self, sigma: float, size: int = 3) -> np.ndarray:
        return self.rng.normal(0.0, sigma, size)

    def true_specific_force(self) -> np.ndarray:
        """Body-frame specific force f_b = R^T (a_n - g_n), noise free."""
        a_b = self.drone.get_acceleration(self._probe).components
        return a_b - self.drone.rotation_matrix.T @ self._g_n

    def true_body_rates(self) -> np.ndarray:
        return self.drone.get_rotation_speed(self._probe).components

    def imu(self) -> tuple[np.ndarray, np.ndarray]:
        """(gyro, accel) sample: truth + bias + white noise."""
        n = self.noise
        scale = 1.0 / np.sqrt(self._imu_dt)
        gyro = self.true_body_rates() + n.gyro_bias + self._white(n.sigma_g * scale)
        accel = self.true_specific_force() + n.accel_bias + self._white(n.sigma_a * scale)
        return gyro, accel

    def gps(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.noise
        return (
            self.drone.position + self._white(n.sigma_gps_pos),
            self.drone.velocity + self._white(n.sigma_gps_vel),
        )

    def baro(self) -> float:
        return float(self.drone.position[2] + self._white(self.noise.sigma_baro, 1)[0])

    def mag(self) -> np.ndarray:
        m_b = self.drone.rotation_matrix.T @ self.noise.mag_reference
        return m_b + self._white(self.noise.sigma_mag)
