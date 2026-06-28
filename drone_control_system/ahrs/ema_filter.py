from __future__ import annotations
import numpy as np


class EMAFilter:
    """Exponential Moving Average filter for a 3-D reference vector.

    The internal estimate is maintained in the navigation frame so it remains
    stable as the drone rotates. Each call to update() rotates the body-frame
    measurement into nav, applies the EMA, and returns the estimate back in
    the body frame.

    Parameters
    ----------
    alpha : float
        EMA smoothing coefficient in (0, 1].  alpha = 2 / (N + 1) where N
        is the effective window length in samples.
    initial_value : ndarray, shape (3,), optional
        Starting nav-frame estimate. Defaults to the zero vector.
    """

    def __init__(self, alpha: float, initial_value: np.ndarray | None = None) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._estimate_nav = (
            np.array(initial_value, dtype=float)
            if initial_value is not None
            else np.zeros(3)
        )

    @classmethod
    def from_window_seconds(
        cls,
        window_s: float,
        dt: float,
        initial_value: np.ndarray | None = None,
    ) -> EMAFilter:
        """Create an EMAFilter from a window length in seconds.

        Computes alpha = 2 / (N + 1) where N = window_s / dt.
        """
        N = window_s / dt
        alpha = 2.0 / (N + 1.0)
        return cls(alpha, initial_value)

    def update(self, measurement_body: np.ndarray, R_nb: np.ndarray) -> np.ndarray:
        """Update the filter with a new body-frame measurement.

        Parameters
        ----------
        measurement_body : ndarray, shape (3,)
            Raw sensor reading in the body frame.
        R_nb : ndarray, shape (3,3)
            Current rotation matrix (nav-to-body), so R_nb.T rotates body->nav.

        Returns
        -------
        reference_body : ndarray, shape (3,)
            Filtered reference vector in the body frame.
        """
        v_nav = R_nb.T @ measurement_body
        self._estimate_nav = self._alpha * v_nav + (1.0 - self._alpha) * self._estimate_nav
        return R_nb @ self._estimate_nav

    @property
    def estimate_nav(self) -> np.ndarray:
        return self._estimate_nav.copy()
