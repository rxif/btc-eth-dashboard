import numpy as np
import pandas as pd
from pathlib import Path
from pricing.black76 import implied_vol_vec, vega

def load_snapshot(path:str) -> pd.DataFrame :
    
    df = pd.read_parquet(path)
    
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["T"] = (df["expiry"] - df["snapshot_ts"]).dt.total_seconds()/(365*24*3600)
    
    return df

def add_implied_vol(df: pd.DataFrame) -> pd.DataFrame :
    F = df["underlying_price"]

    bid = df["bid_price"].where(df["bid_price"] > 0)  
    ask = df["ask_price"].where(df["ask_price"] > 0)
    mid = (bid + ask) / 2                             

    df["usd_bid"] = bid * F
    df["usd_ask"] = ask * F
    df["usd_mid"] = mid * F
    df["usd_mark"] = df["mark_price"] * F

    k = df["strike"].to_numpy()
    T = df["T"].to_numpy()
    is_call = (df["option_type"]=="C").to_numpy()
    Fnp = F.to_numpy()

    for name in ["bid" , "ask", "mid", "mark"]:
        price = df[f"usd_{name}"].to_numpy()
        df[f"iv_{name}"] = implied_vol_vec(price,Fnp,k,T,is_call)

    df["iv_ref"] = df["mark_iv"]/100

    return df 

def add_quality_column(df:pd.DataFrame) -> pd.DataFrame :
    
    df["spread_iv"] = df["iv_ask"] - df["iv_bid"]
    
    F = df["underlying_price"].to_numpy()
    k = df["strike"].to_numpy()
    T = df["T"].to_numpy()
    is_call = (df["option_type"]=="C").to_numpy()
    sigma = df["iv_mark"].to_numpy()
    df["vega"] = vega(F,k,T,sigma)

    df["log_m"] = np.log(df["strike"]/df["underlying_price"])
    df["otm"] = np.where(is_call, k>=F, k<=F)

    return df 


def clean_chain(df:pd.DataFrame, max_spread:float=0.05, min_vega:float=200.0, min_T:float=1/365) -> pd.DataFrame :
    
    clean = df[df["otm"] & df["iv_mid"].notna() & (df["T"]>min_T) & (df["vega"]>min_vega) & (df["spread_iv"]<max_spread)].copy()

    return clean 
