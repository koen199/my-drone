"""Shared fixtures describing the reference quadrotor used across the tests."""
import numpy as np
import pytest

from simulations.control_system.geometry import QuadrotorGeometry

MASS = 1.2
INERTIA = np.diag([0.011, 0.011, 0.021])


@pytest.fixture
def geometry() -> QuadrotorGeometry:
    """A ~1.2 kg quadrotor with deliberately unequal arms.

    l1 != l2 keeps the tests honest: a square frame would hide any confusion
    between the roll and pitch moment arms.
    """
    return QuadrotorGeometry(
        roll_arm=0.17,
        pitch_arm=0.13,
        thrust_coefficient=6.0e-6,
        drag_coefficient=9.6e-8,
        min_motor_speed=0.0,
        max_motor_speed=1200.0,
    )


@pytest.fixture
def mass() -> float:
    return MASS


@pytest.fixture
def inertia() -> np.ndarray:
    return INERTIA.copy()
