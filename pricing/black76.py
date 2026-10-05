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

def Newton(prix_usd:float, F:float, k:float, T:float, is_call:bool, max_it:float=50, tol:float=1e-8) -> float:
    
    sigma = 0.5
    for i in range(max_it):
        
        ecart = bs_price(F, k, T, sigma, is_call) - prix_usd
        if np.abs(ecart) < tol:
            return sigma

        v = vega(F, k, T, sigma)
        if not np.isfinite(v) or v<1e-12:
            return np.nan
        
        sigma = sigma - ecart/v
        
        if not np.isfinite(sigma) or sigma<1e-4 or sigma>5:
            return np.nan
    
    return np.nan

def _brent(prix_usd: float, F:float, k:float, T:float, is_call:bool, tol:float=1e-8) -> float:
    f= lambda s: bs_price(F,k,T,s,is_call)
    try:
        return brentq(f, 1e-4, 5, xtol=tol)
    except ValueError:
        return np.nan

def implied_vol(prix_usd:float, F:float, k:float, T:float, is_call) -> float:

    if T<=0 or prix_usd<=0:
        return np.nan
    if is_call:
        born_inf, born_max = max(F-k, 0), F
    else:
         born_inf, born_max = max(k-F, 0), k

    if prix_usd<born_inf or prix_usd>born_max:
        return np.nan
    
    sigma = Newton(prix_usd, F, k, T, is_call)
    if np.isnan(sigma):
        sigma = _brent(prix_usd, F, k, T, is_call)
    return sigma

