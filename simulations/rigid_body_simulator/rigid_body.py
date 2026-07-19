"""
Rigid body dynamics as described in docs/rigid_body_simulator/rigid_body_simulator.tex.

State vector (flat, 13 elements, all expressed in the world/navigation frame n):
    x = [r(0:3), v(3:6), p(6:10), omega(10:13)]
where r is the world position of the center of mass, v its linear velocity,
p the orientation quaternion (scalar-first, n<-b) and omega the angular
velocity in the world frame.

Forces and torques may be expressed in any CoordinateSystem; they are
re-expressed in the world frame with the orientation of the state being
evaluated, so body-fixed forces rotate correctly inside RK4 substeps.
"""
from __future__ import annotations

import numpy as np

from .frames import CoordinateSystem, Force, Torque, Vector
from .quaternion import (
    quaternion_derivative,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
)
from .solvers import Integratable

_IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


class RigidBody(Integratable):
    """A rigid body advanced by a Solver and probed like a sensor platform.

    Parameters
    ----------
    inertia_body : ndarray, shape (3, 3)
        Inertia tensor about the center of mass, expressed in the body frame.
        Must be symmetric positive-definite.
    mass : float
        Total mass in kg, must be > 0.
    cog_position : Vector
        Position of the center of mass relative to the body origin. Must be
        expressed in the BODY frame (the CoG is fixed in the body).
    initial_position : Vector, optional
        World position of the BODY ORIGIN (not the CoM); converted internally
        to the CoM state position. Defaults to the world origin.
    initial_velocity : Vector, optional
        World-frame velocity of the center of mass. Defaults to zero.
    initial_orientation : ndarray, shape (4,), optional
        Orientation quaternion [w, x, y, z] (n<-b). Defaults to identity.
    initial_angular_velocity : Vector, optional
        World-frame angular velocity. Defaults to zero.
    """

    def __init__(
        self,
        inertia_body: np.ndarray,
        mass: float,
        cog_position: Vector,
        initial_position: Vector | None = None,
        initial_velocity: Vector | None = None,
        initial_orientation: np.ndarray | None = None,
        initial_angular_velocity: Vector | None = None,
    ) -> None:
        inertia_body = np.asarray(inertia_body, dtype=float)
        if inertia_body.shape != (3, 3):
            raise ValueError(f"inertia_body must have shape (3, 3), got {inertia_body.shape}")
        if not np.allclose(inertia_body, inertia_body.T, atol=1e-10):
            raise ValueError("inertia_body must be symmetric")
        if np.any(np.linalg.eigvalsh(inertia_body) <= 0.0):
            raise ValueError("inertia_body must be positive-definite")
        if mass <= 0.0:
            raise ValueError(f"mass must be > 0, got {mass}")
        if cog_position.coordinate_system is not CoordinateSystem.BODY:
            raise ValueError("cog_position must be expressed in the BODY frame")

        self._inertia_body = inertia_body.copy()
        self._mass = float(mass)
        self._cog_body = cog_position.components.copy()

        p = _IDENTITY_QUATERNION if initial_orientation is None else np.asarray(
            initial_orientation, dtype=float
        )
        if p.shape != (4,):
            raise ValueError(f"initial_orientation must have shape (4,), got {p.shape}")
        p = quaternion_normalize(p)

        origin_n = self._require_world(initial_position, "initial_position")
        v = self._require_world(initial_velocity, "initial_velocity")
        omega = self._require_world(initial_angular_velocity, "initial_angular_velocity")

        # The state stores the CoM position; the user supplies the body origin.
        r = origin_n + quaternion_to_rotation_matrix(p) @ self._cog_body

        self._state = np.concatenate([r, v, p, omega])
        self._forces: list[Force] = []
        self._torques: list[Torque] = []

    @staticmethod
    def _require_world(vec: Vector | None, name: str) -> np.ndarray:
        if vec is None:
            return np.zeros(3)
        if vec.coordinate_system is not CoordinateSystem.WORLD:
            raise ValueError(f"{name} must be expressed in the WORLD frame")
        return vec.components.copy()

    # ------------------------------------------------------------------
    # Force interface
    # ------------------------------------------------------------------

    def apply_forces(self, forces: list[Force]) -> None:
        """Replace the active force set.

        Forces persist across solver steps until replaced by the next call or
        removed with clear_forces(); they are held constant over a step.
        """
        self._forces = list(forces)

    def apply_torques(self, torques: list[Torque]) -> None:
        """Replace the active pure-torque set. Same persistence as apply_forces."""
        self._torques = list(torques)

    def clear_forces(self) -> None:
        """Remove all active forces and torques."""
        self._forces = []
        self._torques = []

    # ------------------------------------------------------------------
    # Frame transforms (parameterized by state so they work at RK4 stages)
    # ------------------------------------------------------------------

    def _direction_to_world(self, vec: Vector, R_nb: np.ndarray) -> np.ndarray:
        if vec.coordinate_system is CoordinateSystem.WORLD:
            return vec.components
        return R_nb @ vec.components

    def _point_to_world(self, vec: Vector, r_com: np.ndarray, R_nb: np.ndarray) -> np.ndarray:
        if vec.coordinate_system is CoordinateSystem.WORLD:
            return vec.components
        # Body points are relative to the body origin; the state position is
        # the CoM, offset from the origin by the CoG vector.
        return r_com + R_nb @ (vec.components - self._cog_body)

    def _direction_from_world(
        self, v_n: np.ndarray, coordinate_system: CoordinateSystem, R_nb: np.ndarray
    ) -> np.ndarray:
        if coordinate_system is CoordinateSystem.WORLD:
            return v_n
        return R_nb.T @ v_n

    # ------------------------------------------------------------------
    # Integratable interface
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float)
        if state.shape != (13,):
            raise ValueError(f"state must have shape (13,), got {state.shape}")
        state = state.copy()
        # Integration does not preserve the quaternion's unit length.
        state[6:10] = quaternion_normalize(state[6:10])
        self._state = state

    def compute_state_derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        r, v, p, omega = state[0:3], state[3:6], state[6:10], state[10:13]
        p = quaternion_normalize(p)
        R_nb = quaternion_to_rotation_matrix(p)

        # Linear part: a = sum(F) / m.
        force_total = np.zeros(3)
        torque_total = np.zeros(3)
        for f in self._forces:
            f_n = self._direction_to_world(f.force, R_nb)
            arm = self._point_to_world(f.application_point, r, R_nb) - r
            force_total += f_n
            torque_total += np.cross(arm, f_n)
        for tau in self._torques:
            torque_total += self._direction_to_world(tau.torque, R_nb)
        a = force_total / self._mass

        # Angular part: Euler's equations in the world frame.
        inertia_n = R_nb @ self._inertia_body @ R_nb.T
        alpha = np.linalg.solve(
            inertia_n, torque_total - np.cross(omega, inertia_n @ omega)
        )

        p_dot = quaternion_derivative(p, omega)
        return np.concatenate([v, a, p_dot, alpha])

    # ------------------------------------------------------------------
    # Sensor queries
    # ------------------------------------------------------------------

    def get_acceleration(self, position: Vector) -> Vector:
        """Kinematic acceleration of a body-fixed point, in the input's frame.

        a_p = a_com + alpha x rho + omega x (omega x rho), with rho the world
        vector from the CoM to the point. This is the coordinate acceleration;
        modeling an IMU's specific force (gravity subtraction, noise) belongs
        to a future sensor layer.
        """
        state = self._state
        r, omega = state[0:3], state[10:13]
        R_nb = quaternion_to_rotation_matrix(state[6:10])
        derivative = self.compute_state_derivative(0.0, state)
        a_com, alpha = derivative[3:6], derivative[10:13]

        rho = self._point_to_world(position, r, R_nb) - r
        a_point = a_com + np.cross(alpha, rho) + np.cross(omega, np.cross(omega, rho))
        return Vector(
            components=self._direction_from_world(a_point, position.coordinate_system, R_nb),
            coordinate_system=position.coordinate_system,
        )

    def get_rotation_speed(self, position: Vector) -> Vector:
        """Angular velocity at a body-fixed point, in the input's frame.

        The angular velocity is uniform across a rigid body, so the position
        only determines the output coordinate system.
        """
        omega = self._state[10:13]
        R_nb = quaternion_to_rotation_matrix(self._state[6:10])
        return Vector(
            components=self._direction_from_world(omega, position.coordinate_system, R_nb),
            coordinate_system=position.coordinate_system,
        )

    # ------------------------------------------------------------------
    # Read-only state accessors
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """World position of the center of mass."""
        return self._state[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """World velocity of the center of mass."""
        return self._state[3:6].copy()

    @property
    def orientation(self) -> np.ndarray:
        """Orientation quaternion [w, x, y, z] (n<-b)."""
        return self._state[6:10].copy()

    @property
    def angular_velocity(self) -> np.ndarray:
        """World-frame angular velocity."""
        return self._state[10:13].copy()

    @property
    def rotation_matrix(self) -> np.ndarray:
        """R_nb derived from the current orientation (v_n = R_nb @ v_b)."""
        return quaternion_to_rotation_matrix(self._state[6:10])
