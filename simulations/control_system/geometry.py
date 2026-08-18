"""
Quadrotor frame geometry and propeller coefficients.

This module is the single source of truth for the motor layout. Both the
mixer (which allocates a desired wrench to motor thrusts) and the propeller
plant model (which turns motor speeds back into forces and torques on the
rigid body) derive their sign conventions from the values here, so the two
can never disagree about which motor rolls the drone which way.

Body frame: x and y are the axes drawn in
docs/control_system/mixer_image_adapted.png and z completes the right-handed
triad, pointing up out of the airframe. Propeller thrust always acts along
+z_b, and world gravity along -z_n, consistent with the z-up world frame used
by the rigid body simulator.

Motor layout, positions given relative to the center of gravity:
    M1 : (+l2, -l1, 0)   spins CCW about +z_b
    M2 : (-l2, -l1, 0)   spins CW
    M3 : (-l2, +l1, 0)   spins CCW
    M4 : (+l2, +l1, 0)   spins CW
so l1 is the roll moment arm (|y| offset) and l2 the pitch moment arm
(|x| offset).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Spin direction of each motor about the body +z axis: +1 CCW, -1 CW.
# Per the conventions section of docs/control_system/control_system.tex,
# motors 1 and 3 spin counterclockwise and motors 2 and 4 clockwise.
SPIN_DIRECTIONS = np.array([1.0, -1.0, 1.0, -1.0])


@dataclass(frozen=True)
class QuadrotorGeometry:
    """Frame dimensions and aerodynamic coefficients of a quadrotor.

    Parameters
    ----------
    roll_arm : float
        l1, the |y| offset of every motor from the CoG in metres.
    pitch_arm : float
        l2, the |x| offset of every motor from the CoG in metres.
    thrust_coefficient : float
        k_t in F_t = k_t * omega^2, in N*s^2/rad^2.
    drag_coefficient : float
        k_d in T_r = k_d * omega^2, in N*m*s^2/rad^2.
    min_motor_speed, max_motor_speed : float
        Actuator limits in rad/s. min_motor_speed is the idle speed, which is
        also the lower thrust limit the mixer clips to.
    """

    roll_arm: float
    pitch_arm: float
    thrust_coefficient: float
    drag_coefficient: float
    min_motor_speed: float = 0.0
    max_motor_speed: float = np.inf

    def __post_init__(self) -> None:
        if self.roll_arm <= 0.0 or self.pitch_arm <= 0.0:
            raise ValueError("roll_arm and pitch_arm must be > 0")
        if self.thrust_coefficient <= 0.0:
            raise ValueError("thrust_coefficient must be > 0")
        if self.drag_coefficient <= 0.0:
            raise ValueError("drag_coefficient must be > 0")
        if self.min_motor_speed < 0.0:
            raise ValueError("min_motor_speed must be >= 0")
        if self.max_motor_speed <= self.min_motor_speed:
            raise ValueError("max_motor_speed must exceed min_motor_speed")

    @property
    def motor_positions(self) -> np.ndarray:
        """(4, 3) motor positions in the body frame, relative to the CoG."""
        l1, l2 = self.roll_arm, self.pitch_arm
        return np.array([
            [+l2, -l1, 0.0],
            [-l2, -l1, 0.0],
            [-l2, +l1, 0.0],
            [+l2, +l1, 0.0],
        ])

    @property
    def spin_directions(self) -> np.ndarray:
        """(4,) spin direction of each motor about +z_b: +1 CCW, -1 CW."""
        return SPIN_DIRECTIONS.copy()

    @property
    def drag_to_thrust_ratio(self) -> float:
        """c = k_d / k_t, the yaw torque produced per newton of thrust."""
        return self.drag_coefficient / self.thrust_coefficient

    @property
    def min_motor_thrust(self) -> float:
        return self.thrust_coefficient * self.min_motor_speed ** 2

    @property
    def max_motor_thrust(self) -> float:
        return self.thrust_coefficient * self.max_motor_speed ** 2

    def thrust_to_speed(self, thrust: np.ndarray | float) -> np.ndarray:
        """Invert F_t = k_t * omega^2, clipped to the actuator speed range.

        Negative thrust requests are treated as zero: a fixed-pitch propeller
        cannot pull.
        """
        thrust = np.maximum(np.asarray(thrust, dtype=float), 0.0)
        speed = np.sqrt(thrust / self.thrust_coefficient)
        return np.clip(speed, self.min_motor_speed, self.max_motor_speed)

    def speed_to_thrust(self, speed: np.ndarray | float) -> np.ndarray:
        """Evaluate F_t = k_t * omega^2."""
        speed = np.asarray(speed, dtype=float)
        return self.thrust_coefficient * speed ** 2

    @property
    def reaction_torque_per_thrust(self) -> np.ndarray:
        """(4,) yaw torque about +z_b each motor exerts per newton of thrust.

        The magnitude is c = k_d / k_t. The sign is OPPOSITE to the spin
        direction: blade drag resists the rotation, so the airframe feels the
        equal-and-opposite reaction, as stated in the mixer section of
        docs/control_system/control_system.tex.

        The mixer's yaw row and the propeller plant model both read this one
        property, so flipping the convention here keeps the control loop and
        the plant consistent with each other.
        """
        return -self.spin_directions * self.drag_to_thrust_ratio

    def reaction_torque(self, speed: np.ndarray | float) -> np.ndarray:
        """Yaw torque each propeller at `speed` exerts on the airframe.

        Equivalent to -spin * k_d * omega^2, written as c * F_t so that the
        sign convention lives in exactly one place.
        """
        return self.reaction_torque_per_thrust * self.speed_to_thrust(speed)
