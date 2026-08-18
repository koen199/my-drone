"""
Propeller plant model: motor speeds -> loads on the rigid body.

This is the actuator half of the loop, the counterpart to the mixer. Given the
commanded motor speeds it produces the Force and Torque objects that
RigidBody.apply_forces / apply_torques consume:

* each propeller pushes along body +z with F_t = k_t omega^2, applied at that
  motor's position, which produces the roll and pitch torques through its
  moment arm without us having to model them explicitly;
* each propeller also drags on the airframe about body z with
  T_r = k_d omega^2, opposing its spin. That one is a pure torque with no
  moment arm, so it is applied through apply_torques.

Motor positions in QuadrotorGeometry are relative to the CENTER OF GRAVITY
(as drawn in the mixer figure), while RigidBody expects body-frame points
relative to the BODY ORIGIN. Pass the same cog_position you gave RigidBody and
this class does the shift for you.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..rigid_body_simulator.frames import CoordinateSystem, Force, Torque, Vector
from .geometry import QuadrotorGeometry


@dataclass
class PropellerLoads:
    """Forces and torques a set of spinning propellers exerts on the body.

    Feed them to the rigid body as::

        drone.apply_forces(loads.forces + [gravity])
        drone.apply_torques(loads.torques)
    """

    forces: list[Force] = field(default_factory=list)
    torques: list[Torque] = field(default_factory=list)


class Propellers:
    """Four fixed-pitch propellers rigidly mounted to the airframe.

    The response is instantaneous: a commanded speed is reached within the
    same step. Real motors have a spin-up time constant of a few tens of
    milliseconds; adding a first-order lag here would be the natural place to
    model it.

    Parameters
    ----------
    geometry : QuadrotorGeometry
        Motor layout and aerodynamic coefficients.
    cog_position : Vector, optional
        Position of the CoG in the BODY frame, i.e. the same vector passed to
        RigidBody. Defaults to the body origin.
    """

    def __init__(self, geometry: QuadrotorGeometry, cog_position: Vector | None = None) -> None:
        if cog_position is None:
            cog_body = np.zeros(3)
        elif cog_position.coordinate_system is not CoordinateSystem.BODY:
            raise ValueError("cog_position must be expressed in the BODY frame")
        else:
            cog_body = cog_position.components.copy()

        self._geometry = geometry
        self._positions_body = geometry.motor_positions + cog_body
        self._speeds = np.zeros(4)

    def set_speed(self, speeds: np.ndarray) -> PropellerLoads:
        """Set the four motor speeds (rad/s) and return the resulting loads.

        Speeds are clipped to the geometry's actuator limits, so commanding an
        unreachable speed silently produces the closest reachable one.
        """
        speeds = np.asarray(speeds, dtype=float)
        if speeds.shape != (4,):
            raise ValueError(f"speeds must have shape (4,), got {speeds.shape}")

        self._speeds = np.clip(
            speeds, self._geometry.min_motor_speed, self._geometry.max_motor_speed
        )
        return self.loads

    @property
    def loads(self) -> PropellerLoads:
        """Loads produced by the propellers at their current speeds."""
        thrusts = self._geometry.speed_to_thrust(self._speeds)
        forces = [
            Force(
                application_point=Vector(
                    components=self._positions_body[i],
                    coordinate_system=CoordinateSystem.BODY,
                ),
                force=Vector(
                    components=[0.0, 0.0, thrusts[i]],
                    coordinate_system=CoordinateSystem.BODY,
                ),
            )
            for i in range(4)
        ]
        reaction = float(np.sum(self._geometry.reaction_torque(self._speeds)))
        torques = [
            Torque(
                torque=Vector(
                    components=[0.0, 0.0, reaction],
                    coordinate_system=CoordinateSystem.BODY,
                )
            )
        ]
        return PropellerLoads(forces=forces, torques=torques)

    @property
    def speeds(self) -> np.ndarray:
        """Current motor speeds in rad/s."""
        return self._speeds.copy()

    @property
    def thrusts(self) -> np.ndarray:
        """Current per-motor thrusts in N."""
        return self._geometry.speed_to_thrust(self._speeds)

    @property
    def geometry(self) -> QuadrotorGeometry:
        return self._geometry
