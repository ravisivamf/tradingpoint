"""Shared indicator maths. One implementation used by the app, the optimizer and the ML pipeline,
so every part of the system sees exactly the same numbers."""
import numpy as np
import pandas as pd


def compute_zero_lag_ema(series, length=70):
    lag = int(np.floor((length - 1) / 2))
    de_lagged = series + (series - series.shift(lag))
    return de_lagged.ewm(span=length, adjust=False).mean()


def compute_atr(df, length=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def compute_adx_engine(df, length=14):
    """Wilder ADX."""
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    alpha = 1.0 / length
    atr = compute_atr(df, length)
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=alpha, adjust=False).mean().rename("ADX")


def compute_rsi(close, length=14):
    """Wilder RSI (the standard definition). NaN during warm-up, 100 when there are no losses."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return rsi.where(~((loss == 0) & gain.notna()), 100.0)


def get_auto_donchian_length(ticker):
    lookup = {"BTCUSD": 55, "ETHUSD": 55, "SPY": 20, "QQQ": 20, "AAPL": 14,
              "TSLA": 10, "NVDA": 10, "MSFT": 20}
    return lookup.get(ticker, 20)


def compute_institutional_pivots(df, pivot_type="Traditional"):
    high, low = float(df["High"].max()), float(df["Low"].min())
    close = float(df["Close"].iloc[-1])
    rng = high - low
    p = (high + low + close) / 3.0
    if pivot_type == "Fibonacci":
        lv = {"R1": p + 0.382 * rng, "S1": p - 0.382 * rng,
              "R2": p + 0.618 * rng, "S2": p - 0.618 * rng,
              "R3": p + rng, "S3": p - rng}
    else:
        lv = {"R1": 2 * p - low, "S1": 2 * p - high,
              "R2": p + rng, "S2": p - rng,
              "R3": high + 2 * (p - low), "S3": low - 2 * (high - p)}
    return {"P": p, **lv}


# ---------------------------------------------------------------- edge-candle Donchian channel
def _confirmed_pivots(df, n, min_prom_atr):
    """Swing highs/lows ('edge' candles where price pulled back). A pivot at bar k is only known at bar k+n (no look-ahead).
    Returns two lists of (pivot_bar, price). Tiny pivots (prominence < min_prom_atr x ATR) are ignored."""
    h, l = df["High"].values, df["Low"].values
    atr = compute_atr(df).values
    N = len(df)
    highs, lows = [], []
    for i in range(N):
        k = i - n
        if k < n or np.isnan(atr[k]):
            continue
        lo_w = max(0, k - 3 * n)
        if h[k] > h[k - n:k].max() and h[k] >= h[k + 1:k + n + 1].max():
            if h[k] - max(l[lo_w:k].min(), l[k + 1:i + 1].min()) >= min_prom_atr * atr[k]:
                highs.append((k, float(h[k]), i))
        if l[k] < l[k - n:k].min() and l[k] <= l[k + 1:k + n + 1].min():
            if min(h[lo_w:k].max(), h[k + 1:i + 1].max()) - l[k] >= min_prom_atr * atr[k]:
                lows.append((k, float(l[k]), i))
    return highs, lows


def auto_edge_donchian(df, fallback_length=20):
    """Data-driven (length, swing strength): the window is ~4x the typical spacing between this instrument's own swing
    pivots on the active timeframe, so a fast stock/timeframe gets a short channel and a slow one a long channel."""
    try:
        hi, lo = _confirmed_pivots(df, 3, 0.5)
        bars = sorted(p[0] for p in hi + lo)
        if len(bars) < 8:
            return fallback_length, 3
        spacing = float(np.median(np.diff(bars)))
        length = int(np.clip(round(spacing * 4), 10, 60))
        return length, int(np.clip(round(length / 8), 2, 5))
    except Exception:
        return fallback_length, 3


def compute_edge_donchian(df, length=20, n=3, min_prom_atr=0.75):
    """Channel anchored on EDGE candles only: upper = highest confirmed swing high inside the window, lower = lowest confirmed
    swing low. The lines are flat steps that touch the one pullback candle that defined them, not every candle (a classic
    rolling max/min hugs each new high/low). Price may close outside = breakout. Columns: DC_Upper, DC_Lower, DC_up_bar, DC_lo_bar."""
    N = len(df)
    highs, lows = _confirmed_pivots(df, n, min_prom_atr)
    up, lo = np.full(N, np.nan), np.full(N, np.nan)
    up_bar, lo_bar = np.full(N, np.nan), np.full(N, np.nan)
    hp = lp = 0                                   # pivots become usable at their confirmation bar
    act_h, act_l = [], []
    for i in range(N):
        while hp < len(highs) and highs[hp][2] <= i:
            act_h.append(highs[hp]); hp += 1
        while lp < len(lows) and lows[lp][2] <= i:
            act_l.append(lows[lp]); lp += 1
        wh = [p for p in act_h if p[0] >= i - length]
        wl = [p for p in act_l if p[0] >= i - length]
        if wh:
            b = max(wh, key=lambda p: p[1]); up[i], up_bar[i] = b[1], b[0]
        elif i:
            up[i], up_bar[i] = up[i - 1], up_bar[i - 1]
        if wl:
            b = min(wl, key=lambda p: p[1]); lo[i], lo_bar[i] = b[1], b[0]
        elif i:
            lo[i], lo_bar[i] = lo[i - 1], lo_bar[i - 1]
    return pd.DataFrame({"DC_Upper": up, "DC_Lower": lo, "DC_up_bar": up_bar, "DC_lo_bar": lo_bar}, index=df.index)
