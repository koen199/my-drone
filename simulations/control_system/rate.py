"""
Inner-loop rate controllers (docs/control_system/control_system.tex, section
"Rate controller").

Two PID loops live here:

* AngularRateController turns a body-rate error into a torque setpoint. The
  PID is tuned in units of angular ACCELERATION, so a gain of 1 /s means "an
  error of 1 rad/s commands 1 rad/s^2". Premultiplying by the body inertia
  tensor converts that to a physical torque:

      T = I ( Kp e + Ki \\int e dt + Kd de/dt ),   e = omega_sp - omega_m

* VerticalSpeedController is the scalar analogue, mapping a vertical speed
  error to a vertical force through Newton's second law:

      F_z = m ( Kp e + Ki \\int e dt + Kd de/dt ),  e = v_z,sp - v_z,m

  Its integrator is preloaded with a fraction of the hover force so the
  vehicle does not dip on takeoff while the integral winds up against gravity.

Both use the diagonal-gain PID from pid.py; see that module for why the
integral is stored in output (acceleration) units.

One consequence of taking de/dt literally, as the spec writes it: a step in
the setpoint steps the error too, so the derivative term produces a one-sample
torque spike ("derivative kick") that briefly rails the motors. It is visible
at every setpoint change in the attitude_steps scenario of
simulations/control_system_simulate.py and is harmless there, but the two
standard cures are available: pass derivative_filter_tau to smooth it, or
switch the D term to act on the measurement rather than the error.
"""
from __future__ import annotations

import numpy as np

from .pid import PID

GRAVITY = 9.81


class AngularRateController:
    """PID from body-rate error to a body-frame torque setpoint.

    Parameters
    ----------
    inertia : ndarray, shape (3, 3)
        Body-frame inertia tensor about the CoG, in kg*m^2.
    kp, ki, kd : float or array_like, shape (3,)
        Diagonal gains, in units of angular acceleration per unit of rate
        error (1/s), per unit of accumulated error (1/s^2), and per unit of
        error rate (dimensionless) respectively.
    integral_limit : float or array_like, optional
        Anti-windup clamp on the integral term, in rad/s^2.
    max_angular_acceleration : float or array_like, optional
        Symmetric clamp on the total commanded angular acceleration, rad/s^2.
    """

    def __init__(
        self,
        inertia: np.ndarray,
        kp,
        ki=0.0,
        kd=0.0,
        integral_limit=None,
        max_angular_acceleration=None,
        derivative_filter_tau: float = 0.0,
    ) -> None:
        inertia = np.asarray(inertia, dtype=float)
        if inertia.shape != (3, 3):
            raise ValueError(f"inertia must have shape (3, 3), got {inertia.shape}")
        self._inertia = inertia.copy()

        output_limit = None
        if max_angular_acceleration is not None:
            limit = np.abs(np.asarray(max_angular_acceleration, dtype=float))
            output_limit = (-limit, limit)

        self._pid = PID(
            kp=kp,
            ki=ki,
            kd=kd,
            size=3,
            integral_limit=integral_limit,
            output_limit=output_limit,
            derivative_filter_tau=derivative_filter_tau,
        )

    def update(self, rate_setpoint, rate_measured, dt: float) -> np.ndarray:
        """Return the body-frame torque setpoint T in N*m."""
        error = np.asarray(rate_setpoint, dtype=float) - np.asarray(rate_measured, dtype=float)
        angular_acceleration = self._pid.update(error, dt)
        return self._inertia @ angular_acceleration

    def reset(self) -> None:
        self._pid.reset()

    @property
    def pid(self) -> PID:
        return self._pid


class VerticalSpeedController:
    """PID from vertical speed error to the required vertical force F_z.

    Parameters
    ----------
    mass : float
        Vehicle mass in kg.
    kp, ki, kd : float
        Gains in units of vertical acceleration per unit of speed error
        (1/s), per unit of accumulated error (1/s^2), and per unit of error
        rate (dimensionless).
    hover_integral_fraction : float
        Fraction of hover thrust preloaded into the integrator at construction
        and on reset(). The spec suggests ~0.8, i.e. the integral term starts
        at 0.8 g of acceleration so only the last 20 % has to be integrated up
        during takeoff.
    integral_limit : float, optional
        Anti-windup clamp on the integral term, in m/s^2. Defaults to 2 g.
    max_vertical_acceleration : float, optional
        Upper clamp on the commanded vertical acceleration in m/s^2. The lower
        clamp is zero: F_z below zero would ask the propellers to pull the
        vehicle down, which fixed-pitch rotors cannot do.
    """

    def __init__(
        self,
        mass: float,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        hover_integral_fraction: float = 0.8,
        integral_limit: float | None = None,
        max_vertical_acceleration: float | None = None,
        gravity: float = GRAVITY,
        derivative_filter_tau: float = 0.0,
    ) -> None:
        if mass <= 0.0:
            raise ValueError(f"mass must be > 0, got {mass}")

        self._mass = float(mass)
        self._gravity = float(gravity)
        self._hover_integral_fraction = float(hover_integral_fraction)

        self._pid = PID(
            kp=kp,
            ki=ki,
            kd=kd,
            size=1,
            integral_limit=2.0 * self._gravity if integral_limit is None else integral_limit,
            output_limit=(0.0, max_vertical_acceleration),
            initial_integral=self._hover_integral_fraction * self._gravity,
            derivative_filter_tau=derivative_filter_tau,
        )

    def update(self, speed_setpoint: float, speed_measured: float, dt: float) -> float:
        """Return the vertical force setpoint F_z in N (world z, positive up)."""
        error = float(speed_setpoint) - float(speed_measured)
        vertical_acceleration = self._pid.update([error], dt)[0]
        return self._mass * float(vertical_acceleration)

    def reset(self) -> None:
        self._pid.reset(self._hover_integral_fraction * self._gravity)

    @property
    def pid(self) -> PID:
        return self._pid
