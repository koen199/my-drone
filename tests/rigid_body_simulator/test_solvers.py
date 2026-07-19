import numpy as np

from simulations.rigid_body_simulator.solvers import ForwardEuler, Integratable, RK4


class _ExponentialDecay(Integratable):
    """dx/dt = -x, exact solution x(t) = x0 * exp(-t)."""

    def __init__(self, x0: float) -> None:
        self._state = np.array([x0])

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def set_state(self, state: np.ndarray) -> None:
        self._state = state.copy()

    def compute_state_derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        return -state


class _HarmonicOscillator(Integratable):
    """x'' = -x as a first-order system; exact solution cos(t) for x0=[1, 0]."""

    def __init__(self) -> None:
        self._state = np.array([1.0, 0.0])

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def set_state(self, state: np.ndarray) -> None:
        self._state = state.copy()

    def compute_state_derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]])


def _integrate_decay(solver_cls, dt: float, t_end: float) -> float:
    obj = _ExponentialDecay(1.0)
    solver = solver_cls()
    solver.register(obj)
    n_steps = round(t_end / dt)
    for _ in range(n_steps):
        solver.step(dt)
    return abs(obj.get_state()[0] - np.exp(-t_end))


def test_forward_euler_single_step():
    obj = _ExponentialDecay(1.0)
    solver = ForwardEuler()
    solver.register(obj)
    solver.step(0.1)
    assert np.isclose(obj.get_state()[0], 1.0 - 0.1, atol=1e-12)


def test_time_advances():
    solver = RK4()
    solver.register(_ExponentialDecay(1.0))
    assert solver.time == 0.0
    solver.step(0.01)
    solver.step(0.01)
    assert np.isclose(solver.time, 0.02, atol=1e-12)


def test_forward_euler_first_order_convergence():
    err_coarse = _integrate_decay(ForwardEuler, 0.01, 1.0)
    err_fine = _integrate_decay(ForwardEuler, 0.005, 1.0)
    ratio = err_coarse / err_fine
    assert 1.8 < ratio < 2.2


def test_rk4_fourth_order_convergence():
    err_coarse = _integrate_decay(RK4, 0.02, 1.0)
    err_fine = _integrate_decay(RK4, 0.01, 1.0)
    ratio = err_coarse / err_fine
    assert 12.0 < ratio < 20.0


def test_rk4_much_more_accurate_than_euler():
    dt = 0.01
    assert _integrate_decay(RK4, dt, 1.0) < 1e-3 * _integrate_decay(ForwardEuler, dt, 1.0)


def test_rk4_harmonic_oscillator_accuracy():
    obj = _HarmonicOscillator()
    solver = RK4()
    solver.register(obj)
    n_steps = 628
    dt = 2.0 * np.pi / n_steps
    for _ in range(n_steps):
        solver.step(dt)
    # After one full period the state should return to [1, 0].
    assert np.allclose(obj.get_state(), [1.0, 0.0], atol=1e-6)


def test_multiple_registered_objects_advance():
    a = _ExponentialDecay(1.0)
    b = _ExponentialDecay(2.0)
    solver = RK4()
    solver.register(a)
    solver.register(b)
    for _ in range(100):
        solver.step(0.01)
    assert np.isclose(a.get_state()[0], np.exp(-1.0), atol=1e-8)
    assert np.isclose(b.get_state()[0], 2.0 * np.exp(-1.0), atol=1e-8)
