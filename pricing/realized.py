import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.deribit.com/api/v2/public"


def fetch_ohlc(currency: str, start: pd.Timestamp, end: pd.Timestamp,
               resolution: str = "1D") -> pd.DataFrame:
    params = {
        "instrument_name": f"{currency}-PERPETUAL",
        "start_timestamp": int(pd.Timestamp(start).timestamp() * 1000),
        "end_timestamp": int(pd.Timestamp(end).timestamp() * 1000),
        "resolution": resolution,
    }
    r = requests.get(f"{BASE_URL}/get_tradingview_chart_data", params=params, timeout=10)
    r.raise_for_status()
    res = r.json()["result"]
    if res.get("status") != "ok":
        raise ValueError(f"Deribit returned status {res.get('status')!r}")
    return pd.DataFrame(
        {k: res[k] for k in ("open", "high", "low", "close", "volume")},
        index=pd.to_datetime(res["ticks"], unit="ms", utc=True),
    )

def log_returns (close:pd.Series)->pd.Series:
    return np.log(close/close.shift(1)).dropna()

def close_to_close_vol (close:pd.Series, period_per_year:int=365)->float:
   return float(log_returns(close).std(ddof=1)*np.sqrt(period_per_year))
    

def parkinson_vol(lo:pd.Series, hi:pd.Series, periode_per_year:int=365)->float:
    hl = np.log(hi/lo)
    return np.sqrt((hl**2).mean()/(4*np.log(2))*periode_per_year)

def rolling_close_to_close(close:pd.Series, window:int=30, period_per_year:int=365)->pd.Series:
    return log_returns(close).rolling(window = window).std(ddof=1)*np.sqrt(period_per_year)

def rolling_parkinson(lo:pd.Series, hi:pd.Series, window:int=30, periode_per_year:int=365)->pd.Series:
    hl = np.log(hi/lo)
    return np.sqrt((hl**2).rolling(window=window).mean()/(4*np.log(2))*periode_per_year)