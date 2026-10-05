import numpy as np
from pricing.black76 import bs_price, vega, implied_vol


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
    """sigma -> price -> sigma should give back sigma."""
    for F, k, T, s in [(100, 100, 1, 0.2), (100, 130, 0.25, 0.8), (100, 70, 2, 0.5)]:
        for is_call in (True, False):
            price = bs_price(F, k, T, s, is_call)
            assert abs(implied_vol(price, F, k, T, is_call) - s) < 1e-6


def test_impossible_cases_return_nan():
    """No valid vol exists: the solver must return nan, not crash."""
    assert np.isnan(implied_vol(-1, 100, 100, 1, True))    # negative price
    assert np.isnan(implied_vol(5, 100, 90, 1, True))      # below intrinsic value (10)
    assert np.isnan(implied_vol(100, 100, 90, 1, True))    # at the upper bound (F)
    assert np.isnan(implied_vol(5, 100, 100, 0, True))     # T = 0