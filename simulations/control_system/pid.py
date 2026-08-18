"""
Elementwise vector PID controller shared by the rate and vertical speed loops.

docs/control_system/control_system.tex writes the rate control law with
diagonal gain matrices,

    u(t) = I ( Kp e + Ki \\int e dt + Kd de/dt ),

and the off-diagonal entries of Kp, Ki and Kd are zero by construction, so a
diagonal matrix is exactly an elementwise vector gain. This class implements
the bracketed part; premultiplying by the inertia tensor (rate loop) or the
mass (vertical loop) is the caller's job. Consequently the PID output has
units of ACCELERATION, which is what makes the gains physically intuitive:
Kp = 1 means "one unit of error commands one unit of acceleration".

The integrator accumulates in output units (it stores Ki * \\int e dt rather
than \\int e dt), which lets the integral be clamped and preloaded directly in
acceleration units without ever dividing by Ki.
"""
from __future__ import annotations

import numpy as np


class PID:
    """Elementwise PID over a fixed-size error vector.

    Parameters
    ----------
    kp, ki, kd : float or array_like
        Proportional, integral and derivative gains. A scalar is broadcast to
        every axis; an array must have length `size`.
    size : int
        Number of independent axes (3 for body rates, 1 for vertical speed).
    integral_limit : float or array_like, optional
        Symmetric anti-windup clamp on the integral TERM, in output units.
    output_limit : tuple, optional
        (low, high) clamp on the total output, in output units. Either bound
        may be None to leave that side unbounded.
    initial_integral : float or array_like
        Initial value of the integral term, in output units. Used to preload
        the vertical loop with the hover force before takeoff.
    derivative_filter_tau : float
        Time constant in seconds of a first-order low-pass on the derivative
        term. Zero (the default) is the unfiltered de/dt of the spec.
    """

    def __init__(
        self,
        kp,
        ki=0.0,
        kd=0.0,
        size: int = 3,
        integral_limit=None,
        output_limit: tuple[float | None, float | None] | None = None,
        initial_integral=0.0,
        derivative_filter_tau: float = 0.0,
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        if derivative_filter_tau < 0.0:
            raise ValueError("derivative_filter_tau must be >= 0")

        self._size = int(size)
        self._kp = self._as_vector(kp, "kp")
        self._ki = self._as_vector(ki, "ki")
        self._kd = self._as_vector(kd, "kd")
        self._integral_limit = (
            None if integral_limit is None else np.abs(self._as_vector(integral_limit, "integral_limit"))
        )
        self._output_limit = output_limit
        self._initial_integral = self._as_vector(initial_integral, "initial_integral")
        self._derivative_filter_tau = float(derivative_filter_tau)

        self._integral = self._initial_integral.copy()
        self._previous_error: np.ndarray | None = None
        self._derivative = np.zeros(self._size)

    def _as_vector(self, value, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return np.full(self._size, float(array))
        if array.shape != (self._size,):
            raise ValueError(f"{name} must be a scalar or have shape ({self._size},), got {array.shape}")
        return array.copy()

    def reset(self, initial_integral=None) -> None:
        """Clear the derivative memory and reset the integral.

        Passing initial_integral overrides the value given to __init__.
        """
        if initial_integral is not None:
            self._initial_integral = self._as_vector(initial_integral, "initial_integral")
        self._integral = self._initial_integral.copy()
        self._previous_error = None
        self._derivative = np.zeros(self._size)

    def update(self, error, dt: float) -> np.ndarray:
        """Advance the controller one step and return the control output.

        The derivative is a backward difference on the error; on the very
        first call after construction or reset() it is zero, since there is no
        previous sample to difference against.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        error = self._as_vector(error, "error")

        proportional = self._kp * error

        self._integral = self._integral + self._ki * error * dt
        if self._integral_limit is not None:
            self._integral = np.clip(self._integral, -self._integral_limit, self._integral_limit)

        if self._previous_error is None:
            raw_derivative = np.zeros(self._size)
        else:
            raw_derivative = (error - self._previous_error) / dt
        if self._derivative_filter_tau > 0.0:
            alpha = dt / (self._derivative_filter_tau + dt)
            self._derivative = self._derivative + alpha * (raw_derivative - self._derivative)
        else:
            self._derivative = raw_derivative
        self._previous_error = error

        output = proportional + self._integral + self._kd * self._derivative
        if self._output_limit is not None:
            low, high = self._output_limit
            output = np.clip(output, low, high)
        return output

    @property
    def integral(self) -> np.ndarray:
        """Current integral term, in output units."""
        return self._integral.copy()

    @property
    def size(self) -> int:
        return self._size
