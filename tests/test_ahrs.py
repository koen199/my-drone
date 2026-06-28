import numpy as np
import pytest
from drone_control_system.ahrs import AHRSFilter


GRAVITY = np.array([0.0, 0.0, -9.81])
MAG_NORTH = np.array([0.0, 1.0, 0.0])


def _make_level_filter(**kwargs):
    f = AHRSFilter(**kwargs)
    f.initialize(
        accel_body=np.array([0.0, 0.0, 9.81]),
        mag_body=np.array([0.0, 1.0, 0.0]),
    )
    return f


def test_initialize_level_drone():
    f = _make_level_filter()
    roll, pitch, yaw = f.euler_angles
    assert np.isclose(roll, 0.0, atol=1e-10)
    assert np.isclose(pitch, 0.0, atol=1e-10)
    assert np.isclose(yaw, 0.0, atol=1e-10)


def test_update_returns_tuple_of_three():
    f = _make_level_filter()
    result = f.update(np.zeros(3), np.array([0.0, 0.0, 9.81]), np.array([0.0, 1.0, 0.0]))
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_step_count_increments():
    f = _make_level_filter()
    assert f.step_count == 0
    f.update(np.zeros(3), np.array([0.0, 0.0, 9.81]), np.array([0.0, 1.0, 0.0]))
    assert f.step_count == 1
    f.update(np.zeros(3), np.array([0.0, 0.0, 9.81]), np.array([0.0, 1.0, 0.0]))
    assert f.step_count == 2


def test_rotation_matrix_property_is_copy():
    f = _make_level_filter()
    R = f.rotation_matrix
    R[0, 0] = 999.0
    assert f.rotation_matrix[0, 0] != 999.0


def test_update_before_initialize_raises():
    f = AHRSFilter()
    with pytest.raises(RuntimeError):
        f.update(np.zeros(3), np.zeros(3), np.zeros(3))


def test_rotation_matrix_before_initialize_raises():
    f = AHRSFilter()
    with pytest.raises(RuntimeError):
        _ = f.rotation_matrix


def test_static_convergence():
    rng = np.random.default_rng(42)
    f = AHRSFilter(dt=0.01, gravity_gain=0.01, mag_gain=0.01,
                   ema_window_accel_s=60.0, ema_window_mag_s=60.0)
    f.initialize(
        accel_body=np.array([0.0, 0.0, 9.81]),
        mag_body=np.array([0.0, 1.0, 0.0]),
    )
    for _ in range(1000):
        noisy_accel = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.05, 3)
        noisy_mag = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.02, 3)
        roll, pitch, yaw = f.update(np.zeros(3), noisy_accel, noisy_mag)

    assert abs(roll) < np.deg2rad(2.0)
    assert abs(pitch) < np.deg2rad(2.0)
    assert abs(yaw) < np.deg2rad(2.0)


def test_rotation_matrix_remains_valid():
    f = _make_level_filter()
    for _ in range(200):
        f.update(np.zeros(3), np.array([0.0, 0.0, 9.81]), np.array([0.0, 1.0, 0.0]))
    R = f.rotation_matrix
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_yaw_tracking():
    # Constant yaw rate of 10 deg/s for 5 seconds; expect yaw to be tracked
    dt = 0.01
    yaw_rate = np.deg2rad(10.0)
    n_steps = 500  # 5 seconds
    gyro = np.array([0.0, 0.0, yaw_rate])
    accel = np.array([0.0, 0.0, 9.81])

    f = AHRSFilter(dt=dt, gravity_gain=0.01, mag_gain=0.05,
                   ema_window_accel_s=60.0, ema_window_mag_s=5.0)
    f.initialize(accel_body=accel, mag_body=MAG_NORTH.copy())

    # Rotate true mag vector along with the drone
    true_yaw = 0.0
    for _ in range(n_steps):
        true_yaw += yaw_rate * dt
        # True mag in body frame: rotate MAG_NORTH by -true_yaw about Z
        c, s = np.cos(-true_yaw), np.sin(-true_yaw)
        mag_body = np.array([c * MAG_NORTH[0] - s * MAG_NORTH[1],
                              s * MAG_NORTH[0] + c * MAG_NORTH[1],
                              MAG_NORTH[2]])
        f.update(gyro, accel, mag_body)

    _, _, estimated_yaw = f.euler_angles
    expected_yaw = true_yaw % (2 * np.pi)
    # Allow 15 deg tolerance after 5s warm-up
    diff = abs(estimated_yaw - expected_yaw)
    diff = min(diff, 2 * np.pi - diff)
    assert diff < np.deg2rad(15.0), f"Yaw error: {np.rad2deg(diff):.1f} deg"
