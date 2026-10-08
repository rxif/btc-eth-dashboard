import numpy as np
import pandas as pd
import pytest

from pricing.black76 import bs_price
from pricing.smile import (
    load_snapshot, add_implied_vol, add_quality_column, clean_chain,
)

SNAP = pd.Timestamp("2026-10-01 08:00", tz="UTC")
EXPIRY = pd.Timestamp("2026-10-31 08:00", tz="UTC")   # 30 days later


def make_df(rows):
    """Build a snapshot-like DataFrame. Each row: (strike, option_type, sigma_mid, half_spread)."""
    F = 80_000.0
    T = 30 / 365
    out = []
    for k, typ, sig, half in rows:
        is_call = typ == "C"
        mid_usd = float(bs_price(F, k, T, sig, is_call))
        bid_usd = float(bs_price(F, k, T, sig - half, is_call))
        ask_usd = float(bs_price(F, k, T, sig + half, is_call))
        out.append({
            "snapshot_ts": SNAP, "expiry": EXPIRY, "strike": float(k),
            "option_type": typ, "underlying_price": F,
            "bid_price": bid_usd / F, "ask_price": ask_usd / F,
            "mark_price": mid_usd / F, "mark_iv": sig * 100,
        })
    return pd.DataFrame(out)


def prepare(rows, T=30 / 365):
    df = make_df(rows)
    df["T"] = T
    return add_quality_column(add_implied_vol(df))


def test_load_snapshot_computes_T(tmp_path):
    df = make_df([(80_000, "C", 0.5, 0.01)])
    df["expiry"] = pd.Timestamp("2026-10-02 08:00", tz="UTC")   # 1 day after snapshot
    path = tmp_path / "snapshot.parquet"
    df.to_parquet(path)
    out = load_snapshot(path)
    assert out["T"].iloc[0] == pytest.approx(1 / 365)


def test_implied_vols_round_trip():
    df = make_df([(70_000, "P", 0.55, 0.01), (90_000, "C", 0.45, 0.01)])
    df["T"] = 30 / 365
    df = add_implied_vol(df)
    assert df["iv_mark"].to_numpy() == pytest.approx([0.55, 0.45], abs=1e-5)
    assert df["iv_mid"].to_numpy() == pytest.approx([0.55, 0.45], abs=5e-4)
    assert df["iv_ref"].to_numpy() == pytest.approx([0.55, 0.45])
    assert (df["iv_bid"] < df["iv_mid"]).all()
    assert (df["iv_ask"] > df["iv_mid"]).all()


def test_missing_bid_gives_nan_mid():
    df = make_df([(70_000, "P", 0.55, 0.01)])
    df["T"] = 30 / 365
    df["bid_price"] = 0.0
    df = add_implied_vol(df)
    assert np.isnan(df["iv_bid"].iloc[0])
    assert np.isnan(df["iv_mid"].iloc[0])
    assert not np.isnan(df["iv_mark"].iloc[0])


def test_log_moneyness_and_otm_flags():
    df = prepare([(80_000, "C", 0.5, 0.01), (90_000, "C", 0.5, 0.01),
                  (90_000, "P", 0.5, 0.01), (70_000, "P", 0.5, 0.01)])
    assert df["log_m"].iloc[0] == pytest.approx(0.0)
    assert df["log_m"].iloc[1] == pytest.approx(np.log(90_000 / 80_000))
    assert df["otm"].tolist() == [True, True, False, True]


def test_vega_is_positive_for_valid_rows():
    df = prepare([(80_000, "C", 0.5, 0.01)])
    assert df["vega"].iloc[0] > 0


def test_clean_chain_filters():
    df = prepare([
        (90_000, "C", 0.50, 0.005),   # OTM, tight spread -> kept
        (70_000, "P", 0.55, 0.005),   # OTM, tight spread -> kept
        (70_000, "C", 0.55, 0.005),   # ITM call -> dropped
        (90_000, "C", 0.50, 0.100),   # OTM, very wide spread -> dropped
    ])
    out = clean_chain(df, max_spread=0.05, min_vega=0.0)
    assert out["strike"].tolist() == [90_000.0, 70_000.0]
    assert out["spread_iv"].max() < 0.05


def test_clean_chain_drops_row_without_bid():
    df = make_df([(90_000, "C", 0.5, 0.005), (91_000, "C", 0.5, 0.005)])
    df["T"] = 30 / 365
    df.loc[1, "bid_price"] = 0.0
    df = add_quality_column(add_implied_vol(df))
    out = clean_chain(df, min_vega=0.0)
    assert out["strike"].tolist() == [90_000.0]


def test_clean_chain_drops_short_maturity():
    df = prepare([(90_000, "C", 0.5, 0.005)], T=0.5 / 365)
    assert len(clean_chain(df, min_vega=0.0)) == 0


def test_clean_chain_drops_low_vega():
    df = prepare([(90_000, "C", 0.5, 0.005)])
    assert len(clean_chain(df, min_vega=1e12)) == 0


def test_clean_chain_returns_independent_copy():
    df = prepare([(90_000, "C", 0.5, 0.005)])
    out = clean_chain(df, min_vega=0.0)
    out["new_col"] = 1
    assert "new_col" not in df.columns