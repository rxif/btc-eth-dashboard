from functools import lru_cache
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from pricing import vrp
from pricing.smile import load_snapshot, add_implied_vol, add_quality_column, clean_chain
from pricing.svi import fit_surface, observed_k_ranges
from pricing.realized import fetch_ohlc, rolling_close_to_close, rolling_parkinson
from pricing.vrp import vrp_summary

ROOT = Path(__file__).resolve().parent.parent


def list_snapshots(currency="BTC"):
    folder = ROOT / "data" / "raw" / f"currency={currency}"
    return sorted(folder.glob("date=*/snapshot_*.parquet"))


_build_cache = {}
_build_lock = threading.Lock()


def _compute_build(path_str):
    df = add_quality_column(add_implied_vol(load_snapshot(path_str)))
    clean = clean_chain(df)
    slices = fit_surface(clean)
    kr = observed_k_ranges(clean, slices) if slices else {}
    return clean, slices, kr


def build(path_str):
    hit = _build_cache.get(path_str)
    if hit is not None:
        return hit
    with _build_lock:
        hit = _build_cache.get(path_str)
        if hit is None:
            hit = _compute_build(path_str)
            _build_cache[path_str] = hit
            if len(_build_cache) > 128:
                _build_cache.pop(next(iter(_build_cache)))
        return hit

@lru_cache(maxsize=128)
def load_raw(path_str):
    return pd.read_parquet(path_str)

@lru_cache(maxsize=256)
def download(currency, end):
    return fetch_ohlc(currency, end - pd.Timedelta(days=120), end)

def compute_vrp(slices, currency, end):
    end = pd.Timestamp(end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    end = end.normalize()
    df = download(currency, end)
    vrp = vrp_summary(slices, df)
    roll_close = rolling_close_to_close(df[:-1]["close"]).dropna()
    roll_park = rolling_parkinson(df[:-1]["high"], df[:-1]["low"]).dropna()
    return vrp, roll_close, roll_park

def k_range_at(T, slices, kr):

    Tn = [t for t, _ in slices]
    j = int(np.searchsorted(Tn, T))
    if j < len(Tn) and Tn[j] == T:
        return kr[T]
    a, b = Tn[max(j - 1, 0)], Tn[min(j, len(Tn) - 1)]
    return max(kr[a][0], kr[b][0]), min(kr[a][1], kr[b][1])

def warm_up(currency="BTC", n=8, step=1, every=60):
    while True:
        try:
            snaps = list_snapshots(currency)
            for p in snaps[::-step][:n]:            
                try:
                    build(str(p))
                    load_raw(str(p))
                except Exception:
                    continue                          
                time.sleep(1)                         
        except Exception:
            pass
        time.sleep(every)

def compute_vrp(slices, currency, end):
    end = pd.Timestamp(end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    end = end.normalize()
    df = download(currency, end)
    summary = vrp_summary(slices, df)
    done = df.iloc[:-1]
    roll_close = rolling_close_to_close(done["close"]).dropna()
    roll_park = rolling_parkinson(done["high"], done["low"]).dropna()
    return summary, roll_close, roll_park