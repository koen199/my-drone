"""
Tests for the error-state Kalman filter of docs/state_estimator/state_estimator.tex.

The Jacobian tests are the important ones: they check the doc's analytic F
and H matrices against finite differences of the nominal propagation and the
measurement functions, perturbing the nominal state with the same (+)
operator the filter uses for injection.
"""
import numpy as np
import pytest

from simulations.math_helper import (
    euler_to_rotation_matrix,
    quaternion_conjugate,
    quaternion_exp,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from simulations.state_estimator import (
    ERROR_STATE_SIZE,
    EskfNoise,
    ErrorStateKalmanFilter,
    NominalState,
)

G = 9.81
G_N = np.array([0.0, 0.0, -G])
M_N = np.array([0.0, 0.22, -0.42])


def _state(roll=0.2, pitch=-0.3, yaw=0.7) -> NominalState:
    return NominalState(
        p=np.array([1.0, -2.0, 3.0]),
        v=np.array([0.5, -0.4, 0.3]),
        q=rotation_matrix_to_quaternion(euler_to_rotation_matrix(roll, pitch, yaw)),
        b_g=np.array([0.01, -0.02, 0.03]),
        b_a=np.array([0.1, -0.05, 0.2]),
    )


def _filter(state: NominalState | None = None, **noise) -> ErrorStateKalmanFilter:
    eskf = ErrorStateKalmanFilter(EskfNoise(**noise))
    eskf.state = state or _state()
    return eskf


def _oplus(x: NominalState, dx: np.ndarray) -> NominalState:
    return NominalState(
        p=x.p + dx[0:3],
        v=x.v + dx[3:6],
        q=quaternion_multiply(x.q, quaternion_exp(dx[6:9])),
        b_g=x.b_g + dx[9:12],
        b_a=x.b_a + dx[12:15],
    )


def _ominus(x1: NominalState, x0: NominalState) -> np.ndarray:
    dq = quaternion_multiply(quaternion_conjugate(x0.q), x1.q)
    if dq[0] < 0.0:
        dq = -dq
    return np.concatenate([
        x1.p - x0.p, x1.v - x0.v, 2.0 * dq[1:], x1.b_g - x0.b_g, x1.b_a - x0.b_a,
    ])


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------

def test_error_transition_matches_finite_differences():
    gyro = np.array([0.4, -0.6, 0.9])
    accel = np.array([0.5, -0.3, 9.9])
    dt = 0.005
    x0 = _state()

    def propagate(x: NominalState) -> NominalState:
        eskf = _filter(x)
        eskf.predict(gyro, accel, dt)
        return eskf.state

    nominal = propagate(x0)
    eps = 1e-6
    F_numeric = np.zeros((ERROR_STATE_SIZE, ERROR_STATE_SIZE))
    for j in range(ERROR_STATE_SIZE):
        dx = np.zeros(ERROR_STATE_SIZE)
        dx[j] = eps
        F_numeric[:, j] = _ominus(propagate(_oplus(x0, dx)), nominal) / eps

    F = _filter(x0).error_transition(gyro, accel, dt)
    # The doc's F is first order in dt; the nominal propagation uses the
    # exact exponential map, so they agree to O(|w|^2 dt^2).
    np.testing.assert_allclose(F, F_numeric, atol=1e-4)


@pytest.mark.parametrize("model", ["mag_model", "accel_model"])
def test_measurement_jacobian_matches_finite_differences(model):
    x0 = _state()
    z0, H = getattr(_filter(x0), model)()
    eps = 1e-7
    H_numeric = np.zeros_like(H)
    for j in range(ERROR_STATE_SIZE):
        dx = np.zeros(ERROR_STATE_SIZE)
        dx[j] = eps
        z1, _ = getattr(_filter(_oplus(x0, dx)), model)()
        H_numeric[:, j] = (z1 - z0) / eps
    np.testing.assert_allclose(H, H_numeric, atol=1e-5)


def test_measurement_models_match_physics():
    x0 = _state()
    eskf = _filter(x0)
    R = quaternion_to_rotation_matrix(x0.q)
    np.testing.assert_allclose(eskf.mag_model()[0], R.T @ M_N)
    np.testing.assert_allclose(eskf.accel_model()[0], -R.T @ G_N + x0.b_a)


# ---------------------------------------------------------------------------
# Initialisation and helpers
# ---------------------------------------------------------------------------

def test_initialize_recovers_tilted_attitude():
    R_true = euler_to_rotation_matrix(np.deg2rad(50.0), np.deg2rad(-35.0), np.deg2rad(120.0))
    eskf = ErrorStateKalmanFilter(EskfNoise())
    eskf.initialize(
        position=np.zeros(3),
        velocity=np.zeros(3),
        accel_body=-R_true.T @ G_N,
        mag_body=R_true.T @ M_N,
    )
    np.testing.assert_allclose(eskf.rotation_matrix, R_true, atol=1e-10)
    assert np.allclose(eskf.state.b_g, 0.0) and np.allclose(eskf.state.b_a, 0.0)


def test_accel_update_is_gated_during_manoeuvres():
    eskf = _filter(_state(0.0, 0.0, 0.0))
    P_before = eskf.P
    assert eskf.update_accel(np.array([0.0, 0.0, 12.0])) is None
    np.testing.assert_array_equal(eskf.P, P_before)
    assert eskf.update_accel(np.array([0.0, 0.0, G])) is not None


# ---------------------------------------------------------------------------
# Filter behaviour on a stationary vehicle
# ---------------------------------------------------------------------------

def _run_stationary(R_true, gyro_bias, accel_bias, noise_scale, duration=30.0, seed=1):
    """Hold the vehicle still at the origin and feed the filter every sensor."""
    rng = np.random.default_rng(seed)
    noise = EskfNoise()
    dt = 0.005
    eskf = ErrorStateKalmanFilter(noise)
    eskf.initialize(np.zeros(3), np.zeros(3), -R_true.T @ G_N + accel_bias, R_true.T @ M_N)

    def white(sigma, size=3):
        return noise_scale * rng.normal(0.0, sigma, size)

    for step in range(int(duration / dt)):
        gyro = gyro_bias + white(noise.sigma_g / np.sqrt(dt))
        accel = -R_true.T @ G_N + accel_bias + white(noise.sigma_a / np.sqrt(dt))
        eskf.predict(gyro, accel, dt)
        if step % 2 == 0:
            eskf.update_accel(accel)
        if step % 4 == 0:
            eskf.update_mag(R_true.T @ M_N + white(noise.sigma_mag))
            eskf.update_baro(white(noise.sigma_baro, 1)[0])
        if step % 20 == 0:
            eskf.update_gps(white(noise.sigma_gps_pos), white(noise.sigma_gps_vel))
    return eskf


def test_noiseless_stationary_filter_stays_at_truth():
    R_true = euler_to_rotation_matrix(0.1, -0.2, 0.5)
    eskf = _run_stationary(R_true, np.zeros(3), np.zeros(3), noise_scale=0.0, duration=5.0)
    state = eskf.state
    np.testing.assert_allclose(state.p, 0.0, atol=1e-9)
    np.testing.assert_allclose(state.v, 0.0, atol=1e-9)
    np.testing.assert_allclose(eskf.rotation_matrix, R_true, atol=1e-9)

    P = eskf.P
    np.testing.assert_allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) > 0.0)


def test_stationary_filter_estimates_observable_biases():
    """Gyro bias (all axes) and vertical accel bias converge when still.

    Horizontal accelerometer bias is indistinguishable from tilt without
    attitude changes, so it is deliberately not asserted here.
    """
    R_true = euler_to_rotation_matrix(0.05, -0.1, 0.8)
    gyro_bias = np.array([0.01, -0.008, 0.005])
    accel_bias = np.array([0.08, -0.05, 0.1])
    eskf = _run_stationary(R_true, gyro_bias, accel_bias, noise_scale=1.0)
    state = eskf.state
    np.testing.assert_allclose(state.b_g, gyro_bias, atol=1e-3)
    assert abs(state.b_a[2] - accel_bias[2]) < 0.02
    # Horizontal bias/tilt ambiguity: the tilt absorbs whatever bias is
    # left over, but the combination seen by the accelerometer is right.
    np.testing.assert_allclose(eskf.accel_model()[0], -R_true.T @ G_N + accel_bias, atol=0.02)
