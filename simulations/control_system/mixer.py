"""
Control allocation (docs/control_system/control_system.tex, section "Mixer").

The mixer maps the control effort vector u = [F, Tx, Ty, Tz]^T onto the four
propeller thrusts f = [Ft1, Ft2, Ft3, Ft4]^T through u = A f, then inverts it:

    f = A^-1 u

and converts each thrust to a motor speed setpoint with F_t = k_t omega^2.

A is built from geometry.py rather than hard-coded, so the allocation and the
propeller plant model share one definition of where each motor sits and which
way it spins. Writing the rows out for the layout in geometry.py:

    F   = Ft1 + Ft2 + Ft3 + Ft4
    Tx  = ( Ft3 + Ft4 - Ft1 - Ft2 ) l1        (torque about x is +y_i * F_i)
    Ty  = ( Ft2 + Ft3 - Ft1 - Ft4 ) l2        (torque about y is -x_i * F_i)
    Tz  = c ( Ft2 + Ft4 - Ft1 - Ft3 )         (reaction opposes the spin)

These agree with the equations in the .tex. One stale spot remains in the
document itself: the Tz row of the printed allocation matrix still reads
[c, -c, c, -c], the sign from before the reaction-torque direction was
corrected, so it contradicts the Tz equation a few lines above it. The
equation is the correct one and is what is implemented here.

Finally, the higher-level loops ask for a WORLD-vertical force F_z while the
propellers can only push along body +z, so the total thrust is scaled up by
the tilt of the vehicle:

    F = F_z / ( cos(phi) cos(theta) )
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import QuadrotorGeometry


@dataclass
class MixerOutput:
    """Result of one allocation.

    Attributes
    ----------
    motor_speeds : ndarray, shape (4,)
        Commanded motor speeds in rad/s, within the geometry's speed limits.
    motor_thrusts : ndarray, shape (4,)
        Thrusts actually achievable, i.e. after clipping. These are what the
        propellers will deliver.
    requested_thrusts : ndarray, shape (4,)
        Thrusts before clipping, useful for detecting saturation margins.
    total_thrust : float
        F, the body-axis thrust the allocation was solved for.
    saturated : bool
        True when any motor was clipped, meaning the achieved wrench differs
        from the requested one.
    """

    motor_speeds: np.ndarray
    motor_thrusts: np.ndarray
    requested_thrusts: np.ndarray
    total_thrust: float
    saturated: bool


class Mixer:
    """Allocates [F_z, Tx, Ty, Tz] to four motor speed setpoints.

    Parameters
    ----------
    geometry : QuadrotorGeometry
        Motor layout and aerodynamic coefficients.
    min_tilt_cosine : float
        Floor on cos(phi) cos(theta) used in the tilt compensation, so that a
        vehicle tilted past ~60 deg (the default 0.5) asks for a bounded
        thrust instead of diverging towards infinity at 90 deg.
    """

    def __init__(self, geometry: QuadrotorGeometry, min_tilt_cosine: float = 0.5) -> None:
        if not 0.0 < min_tilt_cosine <= 1.0:
            raise ValueError(f"min_tilt_cosine must be in (0, 1], got {min_tilt_cosine}")

        self._geometry = geometry
        self._min_tilt_cosine = float(min_tilt_cosine)
        self._allocation_matrix = self._build_allocation_matrix(geometry)
        self._allocation_inverse = np.linalg.inv(self._allocation_matrix)

    @staticmethod
    def _build_allocation_matrix(geometry: QuadrotorGeometry) -> np.ndarray:
        """Assemble A such that [F, Tx, Ty, Tz]^T = A @ f.

        Each column is the wrench produced by one newton of thrust at that
        motor: a unit force along +z_b at position r contributes
        r x [0, 0, 1] = [r_y, -r_x, 0] of torque, plus the propeller's
        reaction torque about z.
        """
        positions = geometry.motor_positions
        return np.vstack([
            np.ones(4),
            positions[:, 1],
            -positions[:, 0],
            geometry.reaction_torque_per_thrust,
        ])

    def total_thrust(self, vertical_force: float, roll: float, pitch: float) -> float:
        """Scale a world-vertical force F_z up to a body-axis thrust F."""
        tilt_cosine = max(np.cos(roll) * np.cos(pitch), self._min_tilt_cosine)
        return float(vertical_force) / tilt_cosine

    def mix(
        self,
        vertical_force: float,
        torque: np.ndarray,
        roll: float = 0.0,
        pitch: float = 0.0,
    ) -> MixerOutput:
        """Allocate a vertical force and body torque to motor speed setpoints.

        Saturation is handled by clipping each motor thrust independently into
        the achievable range. That preserves the total thrust reasonably well
        but distorts the torque when a motor runs out of authority; a
        prioritised scheme that sacrifices thrust to keep attitude control
        would be the next refinement.
        """
        torque = np.asarray(torque, dtype=float)
        if torque.shape != (3,):
            raise ValueError(f"torque must have shape (3,), got {torque.shape}")

        thrust = self.total_thrust(vertical_force, roll, pitch)
        effort = np.array([thrust, torque[0], torque[1], torque[2]])
        requested = self._allocation_inverse @ effort

        clipped = np.clip(
            requested, self._geometry.min_motor_thrust, self._geometry.max_motor_thrust
        )
        speeds = self._geometry.thrust_to_speed(clipped)

        return MixerOutput(
            motor_speeds=speeds,
            motor_thrusts=self._geometry.speed_to_thrust(speeds),
            requested_thrusts=requested,
            total_thrust=thrust,
            saturated=bool(np.any(np.abs(clipped - requested) > 1e-12)),
        )

    def wrench(self, motor_thrusts: np.ndarray) -> np.ndarray:
        """Forward map u = A f: the wrench produced by a set of thrusts."""
        return self._allocation_matrix @ np.asarray(motor_thrusts, dtype=float)

    @property
    def allocation_matrix(self) -> np.ndarray:
        return self._allocation_matrix.copy()

    @property
    def geometry(self) -> QuadrotorGeometry:
        return self._geometry
