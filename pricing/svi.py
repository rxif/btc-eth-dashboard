import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def svi(k:float, a:float, b:float, rho:float, m:float, sigma:float )-> float:
    k = np.asarray(k,dtype=float)
    return a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))

def fit_svi(k:float,w:float)-> tuple[float, float, float, float, float]:
    k = np.asarray(k, dtype=float)
    w = np.asarray(w,dtype=float)

    def reside(x):
        return svi(k, *x)-w

    lo = [-0.1,   1e-6, -0.999, k.min() - 1.0, 1e-4]
    hi = [w.max(), 10.0,  0.999, k.max() + 1.0, 1.0]

    best = None
    m0 = k[np.argmin(w)]
    for b0 in (0.05, 0.2, 0.5):
        for rho in (-0.5, 0.0, 0.5):
            x0 = (w.min(),b0, rho, m0, 0.1)
            x0 = np.clip(x0, lo, hi)
            r = least_squares(reside, x0, bounds = (lo,hi))
            if best is None or r.cost < best.cost:
                best = r 
    return tuple(best.x)

def fit_surface(clean:pd.DataFrame, min_point:int=7) -> list[tuple[float, tuple[float, float, float, float, float]]]:
    list_params = []
    expiry = sorted(clean["expiry"].unique())
    for exp in expiry:
        df = clean[clean["expiry"]==exp]
        if len(df)<min_point:
            continue 
        k = df["log_m"].to_numpy()
        w = (df["T"]*df["iv_mid"]**2).to_numpy()
        params = fit_svi(k, w)
        T =  df["T"].iloc[0]
        list_params.append((T,params))
    return list_params

def observed_k_ranges(clean:pd.DataFrame, slices:list[tuple[float, tuple[float, float, float, float, float]]], margin:float=0.1):
    ends = {}
    for T, _ in slices:
        k = clean.loc[np.isclose(clean["T"], T), "log_m"]
        width = k.max() - k.min()
        ends[T]= (k.min() - margin * width, k.max() + margin * width)
    return ends

def butterfly_density(k:float, a:float, b:float, rho:float, m:float, s:float) -> float:
    k = np.asarray(k, dtype=float)
    R = np.sqrt((k - m) ** 2 + s ** 2)
    w = a + b * (rho * (k - m) + R)
    w1 = b * (rho + (k - m) / R)
    w2 = b * s ** 2 / R ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2

def butterfly_min_on_range(params, k_range, n:int=200):
    grid = np.linspace(k_range[0], k_range[1], n)
    return float(butterfly_density(grid, *params).min())

def check_calendar(slices:list[tuple[float, tuple[float, float, float, float, float]]], k_ranges, n:int=100):
    slices = sorted(slices, key=lambda x: x[0])
    c = []
    for (T1, p1), (T2, p2) in zip(slices[:-1], slices[1:]):
        lo = max(k_ranges[T1][0], k_ranges[T2][0])
        hi = min(k_ranges[T1][1], k_ranges[T2][1])
        if lo >= hi:
            c.append((T1, T2, float("nan")))
            continue
        grid = np.linspace(lo, hi, n)
        gap = svi(grid, *p2) - svi(grid, *p1)
        c.append((T1, T2, float(gap.min())))
    return c

def surface_iv(k:float, T:float, slices)->float:
    k = np.asarray(k, dtype=float)
    slices = sorted(slices, key=lambda x: x[0])
    Ts = np.array([t for t, _ in slices])
    assert np.all(np.diff(Ts) > 0), "slices must contain every T only once"

    if T < Ts[0] or T > Ts[-1]:
        return np.full(k.shape, np.nan)

    i = np.searchsorted(Ts, T)
    if T==Ts[i]:
        w = svi(k, *slices[i][1])
        return np.sqrt(w/T)
    w1 = svi(k, *slices[i-1][1])
    w2 = svi(k, *slices[i][1])
    alpha = (T-Ts[i-1])/(Ts[i]-Ts[i-1])
    w = (1-alpha)*w1 + alpha*w2
    return np.sqrt(w/T)