"""
Coordinate systems and the value types used to describe forces and torques.

A Vector carries its own coordinate system so the rigid body can accept
quantities expressed in any frame and transform them internally:
    - WORLD: the non-inertial world/navigation frame (n), fixed to the world.
    - BODY: the body-fixed frame (b), with its origin fixed somewhere on the
      drone. Body-frame points are relative to the body origin, not the CoG.
"""
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class CoordinateSystem(StrEnum):
    BODY = "BODY"
    WORLD = "WORLD"


@dataclass
class Vector:
    """A 3-element vector tagged with the coordinate system it is expressed in.

    Depending on context a Vector is either a free vector (force, angular
    velocity: rotation only) or a point (application point, sensor position:
    rotation + translation).
    """

    components: np.ndarray
    coordinate_system: CoordinateSystem

    def __post_init__(self) -> None:
        components = np.asarray(self.components, dtype=float)
        if components.shape != (3,):
            raise ValueError(f"Vector components must have shape (3,), got {components.shape}")
        self.components = components.copy()
        self.coordinate_system = CoordinateSystem(self.coordinate_system)


@dataclass
class Force:
    """A force and the point it is applied at, each expressible in any frame.

    application_point is a point (BODY: relative to the body origin, WORLD:
    absolute); force is a free vector.
    """

    application_point: Vector
    force: Vector


@dataclass
class Torque:
    """A pure torque acting on the body, expressible in any frame."""

    torque: Vector
