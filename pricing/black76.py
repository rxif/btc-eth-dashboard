from math import nan
import numpy as np
from numpy._core.umath import NAN
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(F:float, k:float, T:float, sigma:float, is_call: bool) -> float:
    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(F/k) + 0.5*sigma_sqrt_T**2)/sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    call = F*norm.cdf(d1) - k*norm.cdf(d2)
    put = k*norm.cdf(-d2) - F*norm.cdf(-d1)
    return np.where(is_call, call, put)

def vega(F:float, k:float, T:float, sigma:float) -> float : 
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F/k) + 0.5*T*sigma**2)/(sigma*sqrt_T)
    return F * norm.pdf(d1) * sqrt_T

def newton_vec(prix_usd:float, F:float, k:float, T, is_call:bool, tol:float=1e-8, max_iter:int=50) -> float:
    s = np.full(np.shape(prix_usd), 0.5)
    with np.errstate(all="ignore"):
        for _ in range(max_iter):
            diff = bs_price(F, k, T, s, is_call) - prix_usd
            if np.all(np.abs(diff) < tol):
                break
            v = vega(F, k, T, s)
            step = np.where(v > 1e-12, diff / v, 0.0)          
            s = np.where(np.abs(diff) < tol, s, s - step)      
            s = np.clip(s, 1e-4, 5)                            
        diff = bs_price(F, k, T, s, is_call) - prix_usd
    return np.where(np.abs(diff) < tol, s, np.nan)


def implied_vol_vec(prix_usd:float, F:float, k:float, T:float, is_call:bool, tol:float=1e-8, max_iter:int=50):
    prix_usd, F, k, T = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (prix_usd, F, k, T))
    )
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), prix_usd.shape)

    lower = np.where(is_call, np.maximum(F - k, 0), np.maximum(k - F, 0))
    upper = np.where(is_call, F, k)
    valid = (T > 0) & (prix_usd > lower) & (prix_usd < upper)

    return np.where(valid, newton_vec(prix_usd, F, k, T, is_call, tol, max_iter), np.nan)
