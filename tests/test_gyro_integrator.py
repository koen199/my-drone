import numpy as np
from drone_control_system.ahrs.gyro_integrator import integrate_gyro


def test_integrate_zero_omega():
    R = np.eye(3)
    assert np.allclose(integrate_gyro(R, np.zeros(3), 0.01), np.eye(3))


def test_integrate_output_shape():
    assert integrate_gyro(np.eye(3), np.array([0.1, 0.0, 0.0]), 0.01).shape == (3, 3)


def test_integrate_single_axis_rotation():
    # Small rotation about z-axis; first-order result should be close to analytic
    omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s about z
    dt = 0.001  # small enough for first-order to be accurate
    R = integrate_gyro(np.eye(3), omega, dt)
    angle = 1.0 * dt
    # R_nb for a yaw of +angle: v_body = R_nb @ v_nav
    # Body-x in nav = [cos, sin, 0], so R_nb row 0 = [cos, -sin, 0]
    R_analytic = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [          0.0,           0.0,  1.0],
    ])
    assert np.allclose(R, R_analytic, atol=1e-5)


def test_integrate_small_angle_approximation_error_grows():
    # First-order approximation error grows with omega*dt
    omega = np.array([0.0, 0.0, 1.0])
    R_small = integrate_gyro(np.eye(3), omega, 0.001)
    R_large = integrate_gyro(np.eye(3), omega, 0.5)

    angle_small = 0.001
    angle_large = 0.5

    R_exact_small = np.array([
        [ np.cos(angle_small), np.sin(angle_small), 0],
        [-np.sin(angle_small), np.cos(angle_small), 0],
        [0, 0, 1],
    ])
    R_exact_large = np.array([
        [ np.cos(angle_large), np.sin(angle_large), 0],
        [-np.sin(angle_large), np.cos(angle_large), 0],
        [0, 0, 1],
    ])

    err_small = np.linalg.norm(R_small - R_exact_small)
    err_large = np.linalg.norm(R_large - R_exact_large)
    assert err_large > err_small
