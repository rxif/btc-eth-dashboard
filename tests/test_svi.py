import numpy as np
import pandas as pd
import pytest

from pricing.svi import (
    svi, fit_svi, fit_surface, butterfly_density, check_calendar, surface_iv,
)

P = (0.04, 0.1, -0.3, 0.0, 0.1)   # a, b, rho, m, s
P2 = (0.08, 0.1, -0.3, 0.0, 0.1)  # same shape, higher level (a bigger)


def test_svi_value_at_k_equals_m():
    assert svi(0.0, *P) == pytest.approx(0.05)


def test_svi_symmetry_when_rho_zero_and_m_zero():
    p = (0.04, 0.1, 0.0, 0.0, 0.1)
    k = np.array([0.1, 0.5, 1.0])
    assert svi(k, *p) == pytest.approx(svi(-k, *p))


def test_svi_vectorized_shape():
    assert svi(np.linspace(-1, 1, 21), *P).shape == (21,)


def test_svi_wing_slopes():
    eps = 1e-3
    a, b, rho, m, s = P
    right = (svi(50 + eps, *P) - svi(50, *P)) / eps
    left = (svi(-50 + eps, *P) - svi(-50, *P)) / eps
    assert right == pytest.approx(b * (1 + rho), rel=1e-3)
    assert left == pytest.approx(-b * (1 - rho), rel=1e-3)


def test_fit_svi_recovers_curve():
    k = np.linspace(-0.4, 0.4, 15)
    w = svi(k, *P)
    fitted = fit_svi(k, w)
    assert np.abs(svi(k, *fitted) - w).max() < 1e-6


def test_fit_svi_with_small_noise():
    rng = np.random.default_rng(0)
    k = np.linspace(-0.4, 0.4, 15)
    w = svi(k, *P) + rng.normal(0, 1e-5, k.size)
    fitted = fit_svi(k, w)
    assert np.abs(svi(k, *fitted) - w).max() < 1e-4


def make_clean(rows):
    """rows: list of (expiry_str, T, n_points)."""
    out = []
    for exp, T, n in rows:
        k = np.linspace(-0.3, 0.3, n)
        w = svi(k, *P) * (T / 0.1)
        for ki, wi in zip(k, w):
            out.append({"expiry": pd.Timestamp(exp, tz="UTC"), "T": T,
                        "log_m": ki, "iv_mid": np.sqrt(wi / T)})
    return pd.DataFrame(out)


def test_fit_surface_skips_expiries_with_few_points():
    clean = make_clean([("2026-11-01", 0.1, 10), ("2026-12-01", 0.2, 3)])
    slices = fit_surface(clean, min_point=7)
    assert len(slices) == 1
    assert slices[0][0] == pytest.approx(0.1)


def test_fit_surface_returns_sorted_T():
    clean = make_clean([("2026-12-01", 0.2, 10), ("2026-11-01", 0.1, 10)])
    Ts = [t for t, _ in fit_surface(clean)]
    assert Ts == sorted(Ts)


def test_butterfly_density_positive_on_example():
    g = butterfly_density(np.linspace(-1.5, 1.5, 601), *P)
    assert g.min() >= 0


def test_butterfly_density_matches_finite_differences():
    k, eps = 0.2, 1e-4
    w1 = (svi(k + eps, *P) - svi(k - eps, *P)) / (2 * eps)
    w2 = (svi(k + eps, *P) - 2 * svi(k, *P) + svi(k - eps, *P)) / eps ** 2
    w = svi(k, *P)
    expected = (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2
    assert butterfly_density(k, *P) == pytest.approx(expected, rel=1e-3)


def test_butterfly_density_negative_on_steep_wings():
    steep = (0.01, 2.0, 0.9, 0.0, 0.05)
    g = butterfly_density(np.linspace(-1.5, 1.5, 601), *steep)
    assert g.min() < 0


def test_check_calendar_ok_and_violation():
    assert check_calendar([(0.1, P), (0.2, P2)])[0][2] > 0
    assert check_calendar([(0.1, P2), (0.2, P)])[0][2] < 0


def test_check_calendar_sorts_input():
    assert check_calendar([(0.2, P2), (0.1, P)])[0][2] > 0


def test_surface_iv_exact_slice():
    slices = [(0.1, P), (0.2, P2)]
    k = np.array([-0.2, 0.0, 0.2])
    for T, p in slices:
        assert surface_iv(k, T, slices) == pytest.approx(np.sqrt(svi(k, *p) / T))


def test_surface_iv_constant_vol_between_slices():
    sig = 0.5
    slices = [(0.1, (sig**2 * 0.1, 0, 0, 0, 0.1)),
              (0.3, (sig**2 * 0.3, 0, 0, 0, 0.1))]
    out = surface_iv(np.array([-0.1, 0.0, 0.1]), 0.2, slices)
    assert out == pytest.approx(sig)


def test_surface_iv_outside_range_is_nan():
    slices = [(0.1, P), (0.2, P2)]
    assert np.isnan(surface_iv(0.0, 0.05, slices)).all()
    assert np.isnan(surface_iv(0.0, 0.5, slices)).all()


def test_surface_iv_unsorted_input():
    a = surface_iv(0.0, 0.15, [(0.1, P), (0.2, P2)])
    b = surface_iv(0.0, 0.15, [(0.2, P2), (0.1, P)])
    assert a == pytest.approx(b)


def test_surface_iv_total_variance_increases_with_T():
    slices = [(0.1, P), (0.2, P2)]
    k = np.linspace(-0.3, 0.3, 7)
    Ts = np.linspace(0.1, 0.2, 6)
    ws = np.array([surface_iv(k, T, slices) ** 2 * T for T in Ts])
    assert (np.diff(ws, axis=0) >= -1e-12).all()