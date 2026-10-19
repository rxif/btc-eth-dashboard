from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pricing.realized import (
    close_to_close_vol,
    fetch_ohlc,
    log_returns,
    parkinson_vol,
    rolling_close_to_close,
    rolling_parkinson,
)


def closes_from_returns(returns, start=100.0):
    return pd.Series(start * np.exp(np.concatenate([[0.0], np.cumsum(returns)])))


# ---------- close-to-close ----------

def test_log_returns_values():
    close = pd.Series([100.0, 110.0, 99.0])
    assert log_returns(close).tolist() == pytest.approx([np.log(1.1), np.log(0.9)])


def test_close_to_close_constant_price_is_zero():
    assert close_to_close_vol(pd.Series([100.0] * 10)) == 0.0


def test_close_to_close_known_value():
    # returns alternate +r, -r (mean exactly 0): sample std = r * sqrt(n / (n - 1))
    r, n = 0.02, 20
    close = closes_from_returns(np.tile([r, -r], n // 2))
    expected = r * np.sqrt(n / (n - 1)) * np.sqrt(365)
    assert close_to_close_vol(close) == pytest.approx(expected)


def test_close_to_close_recovers_sigma_on_simulated_gbm():
    rng = np.random.default_rng(0)
    sigma, ppy, n = 0.5, 365, 50_000
    returns = rng.normal(0.0, sigma / np.sqrt(ppy), n)
    assert close_to_close_vol(closes_from_returns(returns), ppy) == pytest.approx(sigma, rel=0.02)


def test_close_to_close_periods_per_year_scaling():
    close = closes_from_returns(np.tile([0.01, -0.01], 10))
    assert close_to_close_vol(close, 365 * 24) == pytest.approx(
        close_to_close_vol(close, 365) * np.sqrt(24))


# ---------- Parkinson ----------

def test_parkinson_known_value():
    # ln(H/L) = c every day -> sigma = c / sqrt(4 ln 2) * sqrt(365)
    c = 0.04
    high = pd.Series([100.0 * np.exp(c)] * 10)
    low = pd.Series([100.0] * 10)
    assert parkinson_vol(high, low) == pytest.approx(c / np.sqrt(4 * np.log(2)) * np.sqrt(365))


def test_parkinson_zero_range_is_zero():
    p = pd.Series([100.0] * 5)
    assert parkinson_vol(p, p) == 0.0


# ---------- rolling ----------

def test_rolling_close_to_close_last_value_matches_window_estimate():
    rng = np.random.default_rng(1)
    close = closes_from_returns(rng.normal(0, 0.02, 100))
    window = 30
    rolled = rolling_close_to_close(close, window)
    assert rolled.iloc[-1] == pytest.approx(close_to_close_vol(close.iloc[-(window + 1):]))


def test_rolling_lengths():
    rng = np.random.default_rng(2)
    close = closes_from_returns(rng.normal(0, 0.02, 60))
    rolled = rolling_close_to_close(close, 30)
    assert len(rolled) == 60                      # one value per return
    assert rolled.isna().sum() == 29              # first full window ends at the 30th return


def test_rolling_parkinson_last_value_matches_window_estimate():
    rng = np.random.default_rng(3)
    low = pd.Series(100.0 + rng.random(60))
    high = low * np.exp(rng.uniform(0.01, 0.05, 60))
    rolled = rolling_parkinson(high, low, 30)
    assert rolled.iloc[-1] == pytest.approx(parkinson_vol(high.iloc[-30:], low.iloc[-30:]))


# ---------- fetch_ohlc (no network: the HTTP call is mocked) ----------

def _fake_response(status="ok"):
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"result": {
        "status": status,
        "ticks": [1_700_000_000_000, 1_700_086_400_000],
        "open": [1.0, 2.0], "high": [3.0, 4.0], "low": [0.5, 1.5],
        "close": [2.0, 3.0], "volume": [10.0, 20.0],
    }}
    return resp


def test_fetch_ohlc_builds_dataframe_and_queries_the_perpetual():
    with mock.patch("pricing.realized.requests.get", return_value=_fake_response()) as get:
        df = fetch_ohlc("BTC", pd.Timestamp("2026-01-01", tz="UTC"),
                        pd.Timestamp("2026-01-03", tz="UTC"))
    assert get.call_args.kwargs["params"]["instrument_name"] == "BTC-PERPETUAL"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [2.0, 3.0]
    assert str(df.index.tz) == "UTC"


def test_fetch_ohlc_raises_when_status_not_ok():
    with mock.patch("pricing.realized.requests.get", return_value=_fake_response("no_data")):
        with pytest.raises(ValueError):
            fetch_ohlc("BTC", pd.Timestamp("2026-01-01", tz="UTC"),
                       pd.Timestamp("2026-01-03", tz="UTC"))