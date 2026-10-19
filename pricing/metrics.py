from pricing.black76 import delta, gamma
from pricing.svi import surface_iv
import numpy as np
from scipy.optimize import brentq
import pandas as pd

def iv_of_k_at_T(T:float, slices):
    return lambda k : surface_iv(k, T, slices)


def k_at_delta(slices, T:float, delta_target:float, is_call:bool, k_lo:float=-2.0, k_hi:float=2.0)-> float:
    iv_k=iv_of_k_at_T(T,slices)

    def reside(k):
        return delta(1.0, np.exp(k), T,iv_k(k), is_call) - delta_target 

    if reside(k_lo)*reside(k_hi)>0:
        return np.nan
    return brentq(reside,k_lo,k_hi)

def skew_at_atm(slices, T, h:float=1e-4):
    iv_of_k=iv_of_k_at_T(T,slices)
    return (iv_of_k(h)-iv_of_k(-h))/(2*h)

def RR25(T, slices):
    iv_k=iv_of_k_at_T(T,slices)
    kc = k_at_delta(slices, T, 0.25, True) 
    kp = k_at_delta(slices, T, -0.25, False)
    return iv_k(kc)-iv_k(kp)

def BF25(T, slices):
    iv_k=iv_of_k_at_T(T,slices)
    kc = k_at_delta(slices, T, 0.25, True) 
    kp = k_at_delta(slices, T, -0.25, False)
    return (iv_k(kc)+iv_k(kp))/2 - iv_k(0)

def GEX(df:pd.DataFrame)->dict:
    df = df.copy()
    df["T"] = (df["expiry"] - df["snapshot_ts"]).dt.total_seconds() / (365 * 24 * 3600)
    df["iv_ref"] = df["mark_iv"] / 100
    df = df[df["T"] > 0]
    
    GEX_list = {}
    k_list = sorted(df["strike"].unique())
    for k in k_list:
        df_k=df[(df["strike"]==k)]
        F = df_k["underlying_price"].to_numpy()
        strike = df_k["strike"].to_numpy()
        T = df_k["T"].to_numpy()
        sigma = df_k["iv_ref"]
        OI = df_k ["open_interest"].to_numpy()
        sign = np.where(df_k["option_type"]=="C", 1, -1)

        GEX_array = gamma(F,strike,T,sigma)*OI*F**2*0.01*sign
        GEX_list[k] = np.sum(GEX_array)

    return GEX_list 


def smile_table(slices, k_ranges) -> pd.DataFrame:
    rows = []
    for T, _ in slices:
        kc = k_at_delta(slices, T, 0.25, True)
        kp = k_at_delta(slices, T, -0.25, False)
        k_min, k_max = k_ranges[T]

        rows.append({
            "days": T * 365,
            "atm": float(iv_of_k_at_T(T, slices)(0.0)),
            "rr25": float(RR25(T, slices)),
            "bf25": float(BF25(T, slices)),
            "skew": float(skew_at_atm(slices, T)),
            "in_range": bool(k_min <= kp and kc <= k_max),
        })
    return pd.DataFrame(rows)

    



