"""Structure marks (BOS / CHoCH) for the chart - INFORMATIONAL ONLY, no look-ahead, nothing here drives a trade decision.

A mark is drawn only when ALL of these hold:
  1. the level that breaks is a SIGNIFICANT confirmed swing (prominence >= min_swing_atr x ATR), confirmed n bars after it
     formed, so replay and live agree;
  2. the break is a real CLOSE beyond the level by >= disp_atr x ATR, on a candle whose body closes in the break direction
     (wick pokes and doji closes are ignored);
  3. BOS   = continuation: breaks the latest swing high in an uptrend / swing low in a downtrend;
     CHoCH = reversal: a close through the PROTECTED level (the higher-low that launched an uptrend, the lower-high that
     launched a downtrend). Ordinary minor swings can no longer flip the trend;
  4. the first break only establishes the trend (it is labelled BOS, never CHoCH);
  5. at least `cooldown` bars pass between marks (no clusters on consecutive bars).
Columns: event (+1/-1 on the break bar), event_choch (1 = reversal), level (broken price), level_bar (bar where that level
formed), disp (break size in ATR), trend (+1/-1/0 after the bar)."""
import numpy as np
import pandas as pd

from indicators import compute_atr


def compute_structure(df, n=5, disp_atr=0.3, min_swing_atr=1.0, cooldown=3):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    atr = (df["ATR"] if "ATR" in df else compute_atr(df)).values
    N = len(df)
    event, event_choch, trend_arr = np.zeros(N), np.zeros(N), np.zeros(N)
    lvl_out, lbar_out, disp_out = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    sh_px = sl_px = prot_low = prot_high = np.nan
    sh_bar = sl_bar = -1
    sh_open = sl_open = False
    trend, last_ev = 0, -10 ** 9
    for i in range(N):
        a = 0.0 if np.isnan(atr[i]) else atr[i]
        k = i - n
        if k >= n and not np.isnan(atr[k]):
            w0 = max(0, k - 3 * n)
            if h[k] > h[k - n:k].max() and h[k] >= h[k + 1:k + n + 1].max():
                if h[k] - max(l[w0:k].min(), l[k + 1:i + 1].min()) >= min_swing_atr * atr[k]:
                    sh_px, sh_bar, sh_open = h[k], k, True
                    if trend == -1 and k > last_ev and (np.isnan(prot_high) or h[k] < prot_high):
                        prot_high = h[k]                         # a lower high: tighter protected level in a downtrend
            if l[k] < l[k - n:k].min() and l[k] <= l[k + 1:k + n + 1].min():
                if min(h[w0:k].max(), h[k + 1:i + 1].max()) - l[k] >= min_swing_atr * atr[k]:
                    sl_px, sl_bar, sl_open = l[k], k, True
                    if trend == 1 and k > last_ev and (np.isnan(prot_low) or l[k] > prot_low):
                        prot_low = l[k]                          # a higher low: tighter protected level in an uptrend
        if a > 0 and i - last_ev >= cooldown:
            need = disp_atr * a
            bull = sh_open and c[i] > o[i] and c[i] - sh_px >= need
            bear = sl_open and c[i] < o[i] and sl_px - c[i] >= need
            kind = None
            if trend == 1:
                if not np.isnan(prot_low) and c[i] < o[i] and prot_low - c[i] >= need:
                    kind = (-1, True, prot_low, sl_bar)
                elif bull:
                    kind = (1, False, sh_px, sh_bar)
            elif trend == -1:
                if not np.isnan(prot_high) and c[i] > o[i] and c[i] - prot_high >= need:
                    kind = (1, True, prot_high, sh_bar)
                elif bear:
                    kind = (-1, False, sl_px, sl_bar)
            elif bull:
                kind = (1, False, sh_px, sh_bar)
            elif bear:
                kind = (-1, False, sl_px, sl_bar)
            if kind:
                d, is_choch, lvl, lbar = kind
                if d > 0:
                    sh_open = False
                    base = sl_px if not np.isnan(sl_px) else np.nan
                    prot_low = base if (trend != 1 or np.isnan(prot_low)) else (max(prot_low, base) if not np.isnan(base) else prot_low)
                    trend = 1
                else:
                    sl_open = False
                    base = sh_px if not np.isnan(sh_px) else np.nan
                    prot_high = base if (trend != -1 or np.isnan(prot_high)) else (min(prot_high, base) if not np.isnan(base) else prot_high)
                    trend = -1
                last_ev = i
                event[i], event_choch[i], lvl_out[i], lbar_out[i], disp_out[i] = d, float(is_choch), lvl, lbar, abs(c[i] - lvl) / a
        trend_arr[i] = trend
    return pd.DataFrame({"event": event, "event_choch": event_choch, "level": lvl_out, "level_bar": lbar_out,
                         "disp": disp_out, "trend": trend_arr}, index=df.index)


def compute_rsi_divergence(df, n=3, rsi_col="RSI_14", max_gap=60, valid=5, min_rsi_diff=1.0):
    """Regular RSI divergence at confirmed swings (confirmed n bars late -> no look-ahead).
    Bullish: price lower low, RSI higher low (RSI < 55).  Bearish: price higher high, RSI lower high (RSI > 45).
    ev_* = 1 on the bar the divergence is confirmed; bull/bear = 1 for `valid` bars afterwards."""
    h, l, r = df["High"].values, df["Low"].values, df[rsi_col].values
    N = len(df)
    bull, bear, ev_bull, ev_bear = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    px_bull, px_bear = np.full(N, np.nan), np.full(N, np.nan)   # price of the swing that formed the divergence
    pl = ph = None                      # previous pivot: (bar, price, rsi)
    last_bull = last_bear = -10 ** 9
    for i in range(N):
        k = i - n
        if k >= n and not np.isnan(r[k]):
            if l[k] < l[k - n:k].min() and l[k] <= l[k + 1:k + n + 1].min():
                if pl and k - pl[0] <= max_gap and l[k] < pl[1] and r[k] > pl[2] + min_rsi_diff and r[k] < 55:
                    ev_bull[i], last_bull, px_bull[i] = 1, i, l[k]
                pl = (k, l[k], r[k])
            if h[k] > h[k - n:k].max() and h[k] >= h[k + 1:k + n + 1].max():
                if ph and k - ph[0] <= max_gap and h[k] > ph[1] and r[k] < ph[2] - min_rsi_diff and r[k] > 45:
                    ev_bear[i], last_bear, px_bear[i] = 1, i, h[k]
                ph = (k, h[k], r[k])
        bull[i] = 1.0 if i - last_bull <= valid else 0.0
        bear[i] = 1.0 if i - last_bear <= valid else 0.0
    return pd.DataFrame({"bull": bull, "bear": bear, "ev_bull": ev_bull, "ev_bear": ev_bear,
                         "ev_bull_px": px_bull, "ev_bear_px": px_bear}, index=df.index)
