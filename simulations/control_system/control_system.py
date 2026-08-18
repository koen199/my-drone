"""
Full cascaded control system of docs/control_system/control_system.tex.

Signal flow, outermost loop first::

    altitude_sp --> [ Altitude P ] --> v_z,sp --.
                                                 \\
    attitude_sp --> [ Attitude P ] --> omega_sp --\\
                                                   \\
                              [ Vertical speed PID ] --> F_z --.
                              [ Angular rate PID   ] --> T -----+--> [ Mixer ] --> motor speeds

Everything runs at the single rate passed to update(). On real hardware the
outer loops are usually decimated relative to the rate loop; the demo in
simulations/control_system_simulate.py already runs the controller slower than
the physics, and decimating the outer loops further would be a small change
here.

The controller consumes a StateEstimate and knows nothing about the simulator:
in the demo the estimate is read straight off the rigid body (a perfect state
estimator), but an AHRS or complementary filter drops in unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .altitude import AltitudeController
from .attitude import AttitudeController
from .euler import euler_to_rotation_matrix
from .geometry import QuadrotorGeometry
from .mixer import Mixer
from .rate import GRAVITY, AngularRateController, VerticalSpeedController


@dataclass
class Setpoint:
    """Pilot / guidance command.

    The spec's control system stabilises altitude and attitude; horizontal
    position is not closed, so roll and pitch are commanded directly (as they
    would be by a stick or by an outer position loop added later).
    """

    altitude: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def attitude(self) -> tuple[float, float, float]:
        return self.roll, self.pitch, self.yaw


@dataclass
class StateEstimate:
    """Everything the control system needs to know about the vehicle.

    Attributes
    ----------
    altitude : float
        World z position in m, positive up.
    vertical_speed : float
        World z velocity in m/s, positive up.
    roll, pitch, yaw : float
        ZYX Euler angles in radians.
    body_rates : ndarray, shape (3,)
        Angular velocity in the BODY frame, rad/s. This is what a rate gyro
        measures, and what the rate loop closes on.
    """

    altitude: float
    vertical_speed: float
    roll: float
    pitch: float
    yaw: float
    body_rates: np.ndarray

    @property
    def attitude(self) -> tuple[float, float, float]:
        return self.roll, self.pitch, self.yaw


@dataclass
class ControlGains:
    """Tuning of every loop, in the physically intuitive units of the spec."""

    # Outer altitude loop: v_z,sp = kp_altitude * altitude error.
    kp_altitude: float = 1.2
    min_climb_rate: float = -2.0
    max_climb_rate: float = 3.0

    # Inner vertical speed loop, in vertical acceleration per unit error.
    kp_vertical_speed: float = 4.0
    ki_vertical_speed: float = 2.0
    kd_vertical_speed: float = 0.05
    hover_integral_fraction: float = 0.8
    max_vertical_acceleration: float = 2.5 * GRAVITY

    # Attitude loop: omega_sp = kp_attitude * e_R, per body axis.
    kp_attitude: tuple[float, float, float] = (8.0, 8.0, 4.0)
    max_body_rate: tuple[float, float, float] = (6.0, 6.0, 3.0)

    # Rate loop, in angular acceleration per unit error, per body axis.
    kp_rate: tuple[float, float, float] = (40.0, 40.0, 20.0)
    ki_rate: tuple[float, float, float] = (20.0, 20.0, 10.0)
    kd_rate: tuple[float, float, float] = (0.5, 0.5, 0.2)
    rate_integral_limit: tuple[float, float, float] = (20.0, 20.0, 10.0)


@dataclass
class ControlOutput:
    """Actuator command plus the intermediate signals of every loop.

    The intermediates are kept because they are what you actually look at when
    tuning: if the vehicle will not hold altitude you want to know whether the
    vertical speed setpoint, the force, or the allocation is at fault.
    """

    motor_speeds: np.ndarray
    motor_thrusts: np.ndarray
    total_thrust: float
    vertical_force: float
    torque: np.ndarray
    vertical_speed_setpoint: float
    body_rate_setpoint: np.ndarray
    saturated: bool = False


class ControlSystem:
    """The cascaded altitude + attitude controller and its mixer.

    Parameters
    ----------
    mass : float
        Vehicle mass in kg, used to convert commanded vertical acceleration
        into force.
    inertia : ndarray, shape (3, 3)
        Body-frame inertia tensor about the CoG, used to convert commanded
        angular acceleration into torque.
    geometry : QuadrotorGeometry
        Frame layout, handed to the mixer.
    gains : ControlGains, optional
        Tuning. The defaults are sensible for a ~1 kg quadrotor.
    """

    def __init__(
        self,
        mass: float,
        inertia: np.ndarray,
        geometry: QuadrotorGeometry,
        gains: ControlGains | None = None,
        min_tilt_cosine: float = 0.5,
    ) -> None:
        self._gains = gains if gains is not None else ControlGains()
        self._mass = float(mass)

        self._altitude_controller = AltitudeController(
            kp=self._gains.kp_altitude,
            min_climb_rate=self._gains.min_climb_rate,
            max_climb_rate=self._gains.max_climb_rate,
        )
        self._vertical_speed_controller = VerticalSpeedController(
            mass=mass,
            kp=self._gains.kp_vertical_speed,
            ki=self._gains.ki_vertical_speed,
            kd=self._gains.kd_vertical_speed,
            hover_integral_fraction=self._gains.hover_integral_fraction,
            max_vertical_acceleration=self._gains.max_vertical_acceleration,
        )
        self._attitude_controller = AttitudeController(
            kp=np.asarray(self._gains.kp_attitude, dtype=float),
            max_rate=np.asarray(self._gains.max_body_rate, dtype=float),
        )
        self._rate_controller = AngularRateController(
            inertia=inertia,
            kp=np.asarray(self._gains.kp_rate, dtype=float),
            ki=np.asarray(self._gains.ki_rate, dtype=float),
            kd=np.asarray(self._gains.kd_rate, dtype=float),
            integral_limit=np.asarray(self._gains.rate_integral_limit, dtype=float),
        )
        self._mixer = Mixer(geometry, min_tilt_cosine=min_tilt_cosine)

    def update(self, setpoint: Setpoint, estimate: StateEstimate, dt: float) -> ControlOutput:
        """Run one full control cycle and return the motor commands."""
        # Outer loops: position/attitude error -> speed setpoints.
        vertical_speed_setpoint = self._altitude_controller.update(
            setpoint.altitude, estimate.altitude
        )
        body_rate_setpoint = self._attitude_controller.update(
            euler_to_rotation_matrix(*estimate.attitude),
            euler_to_rotation_matrix(*setpoint.attitude),
        )

        # Inner loops: speed errors -> force and torque.
        vertical_force = self._vertical_speed_controller.update(
            vertical_speed_setpoint, estimate.vertical_speed, dt
        )
        torque = self._rate_controller.update(body_rate_setpoint, estimate.body_rates, dt)

        # Allocation: wrench -> motor speeds.
        mixed = self._mixer.mix(vertical_force, torque, estimate.roll, estimate.pitch)

        return ControlOutput(
            motor_speeds=mixed.motor_speeds,
            motor_thrusts=mixed.motor_thrusts,
            total_thrust=mixed.total_thrust,
            vertical_force=vertical_force,
            torque=torque,
            vertical_speed_setpoint=vertical_speed_setpoint,
            body_rate_setpoint=body_rate_setpoint,
            saturated=mixed.saturated,
        )

    def reset(self) -> None:
        """Clear the integrators, restoring the hover preload."""
        self._vertical_speed_controller.reset()
        self._rate_controller.reset()

    @property
    def mixer(self) -> Mixer:
        return self._mixer

    @property
    def gains(self) -> ControlGains:
        return self._gains
