import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pricing.black76 import delta
from pricing.metrics import (
    BF25,
    GEX,
    RR25,
    iv_of_k_at_T,
    k_at_delta,
    skew_at_atm,
)

SIGMA = 0.5
T = 30 / 365

# b = 0 makes the SVI flat: w = a = sigma^2 * T at every k (two slices for interpolation).
FLAT = [
    (0.05, (SIGMA**2 * 0.05, 0.0, 0.0, 0.0, 1.0)),
    (0.50, (SIGMA**2 * 0.50, 0.0, 0.0, 0.0, 1.0)),
]

# Two synthetic SVI slices (a, b, rho, m, s) with negative skew and convex wings.
SKEWED = [
    (0.05, (0.002, 0.020, -0.5, 0.0, 0.15)),
    (0.50, (0.020, 0.200, -0.5, 0.0, 0.30)),
]


# ---------- iv_of_k_at_T ----------

def test_iv_of_k_at_T_matches_slice_on_quoted_maturity():
    a, b, rho, m, s = SKEWED[0][1]
    k = 0.1
    expected = np.sqrt((a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))) / 0.05)
    assert float(iv_of_k_at_T(0.05, SKEWED)(k)) == pytest.approx(expected)


def test_iv_of_k_at_T_flat_is_sigma_between_slices():
    assert float(iv_of_k_at_T(0.2, FLAT)(0.3)) == pytest.approx(SIGMA)


# ---------- k_at_delta ----------

def test_k_at_delta_call_hits_target_delta():
    k = k_at_delta(FLAT, 0.2, 0.25, True)
    assert float(delta(1.0, np.exp(k), 0.2, SIGMA, True)) == pytest.approx(0.25, abs=1e-8)


def test_k_at_delta_put_hits_target_delta():
    k = k_at_delta(FLAT, 0.2, -0.25, False)
    assert float(delta(1.0, np.exp(k), 0.2, SIGMA, False)) == pytest.approx(-0.25, abs=1e-8)


def test_k_at_delta_otm_call_above_forward_otm_put_below():
    assert k_at_delta(FLAT, 0.2, 0.25, True) > 0 > k_at_delta(FLAT, 0.2, -0.25, False)


def test_k_at_delta_matches_closed_form_on_flat_smile():
    expected = 0.5 * SIGMA**2 * 0.2 - SIGMA * np.sqrt(0.2) * norm.ppf(0.25)
    assert k_at_delta(FLAT, 0.2, 0.25, True) == pytest.approx(expected, abs=1e-8)


def test_k_at_delta_returns_nan_when_not_bracketed():
    assert np.isnan(k_at_delta(FLAT, 0.2, 0.25, True, k_lo=-0.001, k_hi=0.001))


# ---------- skew ----------

def test_skew_of_flat_smile_is_zero():
    assert skew_at_atm(FLAT, 0.2) == pytest.approx(0.0, abs=1e-8)


def test_skew_matches_svi_analytic_slope_on_quoted_maturity():
    a, b, rho, m, s = SKEWED[0][1]
    Tq = 0.05
    w0 = a + b * (rho * (0 - m) + np.sqrt(m**2 + s**2))
    w1 = b * (rho + (0 - m) / np.sqrt(m**2 + s**2))
    sigma0 = np.sqrt(w0 / Tq)
    expected = w1 / (2 * sigma0 * Tq)          # d sigma / dk = w'(k) / (2 sigma T)
    assert skew_at_atm(SKEWED, Tq) == pytest.approx(expected, rel=1e-6)


def test_skew_negative_for_negative_rho():
    assert skew_at_atm(SKEWED, 0.2) < 0


# ---------- RR25 / BF25 ----------

def test_rr_bf_zero_on_flat_smile():
    assert RR25(0.2, FLAT) == pytest.approx(0.0, abs=1e-10)
    assert BF25(0.2, FLAT) == pytest.approx(0.0, abs=1e-10)


def test_rr25_negative_for_negative_rho():
    assert RR25(0.2, SKEWED) < 0


def test_bf25_positive_for_convex_smile():
    assert BF25(0.2, SKEWED) > 0


def test_rr_bf_consistent_with_k_at_delta():
    iv = iv_of_k_at_T(0.2, SKEWED)
    kc = k_at_delta(SKEWED, 0.2, 0.25, True)
    kp = k_at_delta(SKEWED, 0.2, -0.25, False)
    assert RR25(0.2, SKEWED) == pytest.approx(float(iv(kc) - iv(kp)))
    assert BF25(0.2, SKEWED) == pytest.approx(float(0.5 * (iv(kc) + iv(kp)) - iv(0.0)))


# ---------- GEX ----------

SNAP = pd.Timestamp("2026-10-16 10:00", tz="UTC")
EXPIRY = pd.Timestamp("2026-11-27 08:00", tz="UTC")


def _row(strike, opt, oi, expiry=EXPIRY):
    return {"snapshot_ts": SNAP, "expiry": expiry, "strike": strike,
            "underlying_price": 100.0, "mark_iv": 50.0,
            "open_interest": oi, "option_type": opt}


def test_gex_call_positive_put_negative():
    out = GEX(pd.DataFrame([_row(100.0, "C", 10.0), _row(120.0, "P", 10.0)]))
    assert out[100.0] > 0
    assert out[120.0] < 0


def test_gex_call_and_put_same_strike_and_oi_cancel():
    out = GEX(pd.DataFrame([_row(100.0, "C", 5.0), _row(100.0, "P", 5.0)]))
    assert out[100.0] == pytest.approx(0.0, abs=1e-12)


def test_gex_scales_linearly_with_open_interest():
    one = GEX(pd.DataFrame([_row(100.0, "C", 1.0)]))[100.0]
    three = GEX(pd.DataFrame([_row(100.0, "C", 3.0)]))[100.0]
    assert three == pytest.approx(3 * one)


def test_gex_ignores_expired_options():
    expired = pd.Timestamp("2026-10-15 08:00", tz="UTC")
    out = GEX(pd.DataFrame([_row(100.0, "C", 5.0, expiry=expired)]))
    assert out == {}


def test_gex_does_not_modify_input_dataframe():
    df = pd.DataFrame([_row(100.0, "C", 1.0)])
    before = df.columns.tolist()
    GEX(df)
    assert df.columns.tolist() == before