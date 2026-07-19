"""
Generic explicit integration schemes and the interface objects must implement
to be advanced by them.

A Solver owns simulation time and a list of registered Integratable objects.
Each step() advances every registered object by dt. The derivative is always
evaluated on a state passed as an argument (never on internal object state), so
multi-stage schemes such as RK4 can probe candidate states without mutating the
object; set_state() is called exactly once per object per step with the final
result.
"""
from abc import ABC, abstractmethod

import numpy as np


class Integratable(ABC):
    """Interface for anything whose state can be advanced by a Solver."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return a flat copy of the current state vector."""

    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        """Store a new state vector. Implementations may re-normalize parts of it."""

    @abstractmethod
    def compute_state_derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return d(state)/dt evaluated at time t for the GIVEN state argument."""


class Solver(ABC):
    """Base class holding registered objects and simulation time."""

    def __init__(self) -> None:
        self._objects: list[Integratable] = []
        self._time: float = 0.0

    def register(self, obj: Integratable) -> None:
        """Add an object to be advanced on every step()."""
        self._objects.append(obj)

    def step(self, dt: float) -> None:
        """Advance all registered objects from t to t + dt."""
        for obj in self._objects:
            obj.set_state(self._advance(obj, self._time, dt))
        self._time += dt

    @property
    def time(self) -> float:
        return self._time

    @abstractmethod
    def _advance(self, obj: Integratable, t: float, dt: float) -> np.ndarray:
        """Return the object's state advanced from t to t + dt."""


class ForwardEuler(Solver):
    """First-order explicit Euler: x_{n+1} = x_n + dt * f(t_n, x_n)."""

    def _advance(self, obj: Integratable, t: float, dt: float) -> np.ndarray:
        x = obj.get_state()
        return x + dt * obj.compute_state_derivative(t, x)


class RK4(Solver):
    """Classic fourth-order Runge-Kutta scheme."""

    def _advance(self, obj: Integratable, t: float, dt: float) -> np.ndarray:
        x = obj.get_state()
        k1 = obj.compute_state_derivative(t, x)
        k2 = obj.compute_state_derivative(t + dt / 2.0, x + (dt / 2.0) * k1)
        k3 = obj.compute_state_derivative(t + dt / 2.0, x + (dt / 2.0) * k2)
        k4 = obj.compute_state_derivative(t + dt, x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
