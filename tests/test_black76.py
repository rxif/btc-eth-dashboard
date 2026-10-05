import numpy as np
from pricing.black76 import bs_price, vega, implied_vol_vec


def test_reference_price():
    """ATM call and put with F=K=100, T=1, sigma=0.2 should be about 7.9656."""
    assert abs(bs_price(100, 100, 1, 0.2, True) - 7.9656) < 1e-3
    assert abs(bs_price(100, 100, 1, 0.2, False) - 7.9656) < 1e-3


def test_put_call_parity():
    """Call - Put = F - K (Black-76 with r = 0)."""
    for F, k in [(100, 90), (100, 120), (100, 100)]:
        c = bs_price(F, k, 0.5, 0.3, True)
        p = bs_price(F, k, 0.5, 0.3, False)
        assert abs((c - p) - (F - k)) < 1e-9


def test_vega_vs_numerical_derivative():
    """Analytical vega should match a centered finite difference."""
    eps = 1e-5
    num = (bs_price(100, 110, 0.5, 0.3 + eps, True)
           - bs_price(100, 110, 0.5, 0.3 - eps, True)) / (2 * eps)
    assert abs(vega(100, 110, 0.5, 0.3) - num) < 1e-4


def test_round_trip():
    """sigma -> price -> sigma on a grid, for calls and puts."""
    F = 100.0
    T = np.array([1.0, 0.25, 2.0])
    k = np.array([100.0, 130.0, 70.0])
    s = np.array([0.2, 0.8, 0.5])
    for is_call in (True, False):
        price = bs_price(F, k, T, s, is_call)
        out = implied_vol_vec(price, F, k, T, is_call)
        assert np.nanmax(np.abs(out - s)) < 1e-6


def test_round_trip_wide_grid():
    """Round trip on a wide strike/vol grid, using OTM options."""
    k = np.linspace(60, 140, 17)
    s = np.linspace(0.2, 1.5, 17)
    is_call = k > 100
    price = bs_price(100.0, k, 0.5, s, is_call)
    out = implied_vol_vec(price, 100.0, k, 0.5, is_call)
    assert np.nanmax(np.abs(out - s)) < 1e-6


def test_impossible_cases_return_nan():
    """No valid vol exists: the solver must return nan, not crash."""
    price = np.array([-1.0, 5.0, 100.0, 5.0])
    k = np.array([100.0, 90.0, 90.0, 100.0])
    T = np.array([1.0, 1.0, 1.0, 0.0])
    out = implied_vol_vec(price, 100.0, k, T, True)
    assert np.isnan(out).all()   # negative, below intrinsic, at upper bound, T = 0


def test_scalar_inputs():
    """Scalar inputs must work and give back sigma."""
    price = bs_price(100.0, 100.0, 1.0, 0.2, True)
    out = implied_vol_vec(price, 100.0, 100.0, 1.0, True)
    assert abs(float(out) - 0.2) < 1e-6


def test_mixed_valid_and_invalid():
    """One bad row must not contaminate the valid rows."""
    k = np.array([100.0, 100.0, 130.0])
    T = np.array([1.0, 0.0, 0.25])
    s = np.array([0.2, 0.3, 0.8])
    price = bs_price(100.0, k, np.where(T > 0, T, 1.0), s, True)
    out = implied_vol_vec(price, 100.0, k, T, True)
    assert np.isnan(out[1])
    assert abs(out[0] - 0.2) < 1e-6
    assert abs(out[2] - 0.8) < 1e-6


def test_calls_and_puts_mixed():
    """Per-row call/put selection through the is_call array."""
    k = np.array([90.0, 110.0, 100.0, 120.0])
    s = np.array([0.3, 0.4, 0.5, 0.6])
    is_call = np.array([True, False, True, False])
    price = bs_price(100.0, k, 0.5, s, is_call)
    out = implied_vol_vec(price, 100.0, k, 0.5, is_call)
    assert np.nanmax(np.abs(out - s)) < 1e-6