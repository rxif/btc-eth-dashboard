from pricing.metrics import iv_of_k_at_T
from pricing.realized import close_to_close_vol, parkinson_vol

def vrp_summary(slices, ohlc, T=30 / 365, window=30):

    done = ohlc.iloc[:-1]                     
    iv = float(iv_of_k_at_T(T, slices)(0.0))
    rv_cc = close_to_close_vol(done["close"].iloc[-(window + 1):])
    rv_pk = parkinson_vol(done["high"].iloc[-window:], done["low"].iloc[-window:])
    return {
        "iv": iv,
        "rv_cc": rv_cc,
        "rv_pk": rv_pk,
        "vrp_cc": iv - rv_cc,
        "vrp_pk": iv - rv_pk,
    }