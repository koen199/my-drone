import numpy as np
import pytest
from drone_control_system.ahrs.ema_filter import EMAFilter


def test_ema_initial_value_used():
    init = np.array([1.0, 2.0, 3.0])
    f = EMAFilter(alpha=0.1, initial_value=init)
    assert np.allclose(f.estimate_nav, init)


def test_ema_output_shape():
    f = EMAFilter(alpha=0.5)
    out = f.update(np.array([1.0, 0.0, 0.0]), np.eye(3))
    assert out.shape == (3,)


def test_ema_converges_to_constant_input():
    target_nav = np.array([0.0, 0.0, -9.81])
    f = EMAFilter(alpha=0.1, initial_value=np.zeros(3))
    for _ in range(500):
        f.update(target_nav, np.eye(3))  # identity R_nb: body == nav
    assert np.allclose(f.estimate_nav, target_nav, atol=1e-3)


def test_ema_alpha_from_window_seconds():
    f = EMAFilter.from_window_seconds(window_s=60.0, dt=0.01)
    expected_alpha = 2.0 / (6000.0 + 1.0)
    assert np.isclose(f._alpha, expected_alpha)


def test_ema_body_frame_round_trip():
    # With identity R_nb, body output equals nav estimate
    init = np.array([0.0, 0.0, -9.81])
    f = EMAFilter(alpha=0.5, initial_value=init)
    out = f.update(init, np.eye(3))
    assert np.allclose(out, f.estimate_nav)


def test_ema_state_in_nav_frame_unaffected_by_rotation():
    # Feed the same physical vector expressed in two different body orientations;
    # the nav-frame estimate should converge to the same value regardless.
    target_nav = np.array([0.0, 0.0, -9.81])

    # Rotation: 90 degrees about z-axis
    R_nb = np.array([
        [0.0,  1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    target_body_rotated = R_nb @ target_nav

    f = EMAFilter(alpha=0.1, initial_value=np.zeros(3))
    for _ in range(500):
        f.update(target_body_rotated, R_nb)

    assert np.allclose(f.estimate_nav, target_nav, atol=1e-3)


def test_ema_raises_on_invalid_alpha():
    with pytest.raises(ValueError):
        EMAFilter(alpha=0.0)
    with pytest.raises(ValueError):
        EMAFilter(alpha=1.5)
