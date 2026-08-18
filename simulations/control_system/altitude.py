"""
Outer-loop altitude controller (docs/control_system/control_system.tex,
section "Altitude Controller").

A proportional law maps vertical position error to a vertical speed setpoint,

    v_z,sp = Kp ( p_z,sp - p_z,m ),

which is then saturated to the vehicle's climb and sink rate limits before
being handed to the inner-loop vertical speed controller. Zero steady-state
error to a step altitude command comes from the integral action of that inner
loop, not from this one.
"""
from __future__ import annotations


class AltitudeController:
    """P controller from altitude error to a saturated vertical speed setpoint.

    Parameters
    ----------
    kp : float
        Proportional position gain in 1/s.
    min_climb_rate : float
        v_z,min, the most negative (sink) speed allowed, in m/s. Must be <= 0.
    max_climb_rate : float
        v_z,max, the fastest climb allowed, in m/s. Must be >= 0.
    """

    def __init__(
        self,
        kp: float,
        min_climb_rate: float = -2.0,
        max_climb_rate: float = 2.0,
    ) -> None:
        if kp < 0.0:
            raise ValueError(f"kp must be >= 0, got {kp}")
        if min_climb_rate > 0.0:
            raise ValueError(f"min_climb_rate must be <= 0, got {min_climb_rate}")
        if max_climb_rate < 0.0:
            raise ValueError(f"max_climb_rate must be >= 0, got {max_climb_rate}")

        self._kp = float(kp)
        self._min_climb_rate = float(min_climb_rate)
        self._max_climb_rate = float(max_climb_rate)

    def update(self, altitude_setpoint: float, altitude: float) -> float:
        """Return the saturated vertical speed setpoint in m/s."""
        vertical_speed_setpoint = self._kp * (altitude_setpoint - altitude)
        return float(
            min(max(vertical_speed_setpoint, self._min_climb_rate), self._max_climb_rate)
        )

    @property
    def kp(self) -> float:
        return self._kp

    @property
    def climb_rate_limits(self) -> tuple[float, float]:
        return self._min_climb_rate, self._max_climb_rate
