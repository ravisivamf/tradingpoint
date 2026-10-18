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


# ---------------------------------------------------------------- Donchian extension BUY setup (auto length)
def donchian_extension_buy(df, base_len=20, min_dist=30, max_back=400, tol_atr=0.5, cooldown=15,
                           pivot_n=5, rally_atr=3.0, reject=True, confirm=True, trend_guard=False, plain=False):
    """Donchian BUY setup measured the way a trader does it by hand, with strict quality filters (fewer, cleaner signals).

    1. Level = lowest Low of the last `base_len` bars (the normal lower Donchian, default 20).
    2. Extend it LEFT (at least min_dist bars back) to the nearest earlier candle that is a REAL support:
         - its Low is at/below the level,
         - it is a confirmed swing low (lowest of pivot_n bars each side),
         - price rallied at least rally_atr x ATR above it afterwards (a genuine bounce, not a flat drift),
         - price never closed below it... i.e. no later Low undercut it (support still intact).
       N = bar distance to that candle becomes the Donchian length (e.g. 20 -> 124).
    3. BUY when lower Donchian(N) stepped UP versus one bar ago (old low rolled out), the current Low sits on the channel edge,
       the current Low is a HIGHER low within tol_atr x ATR of the old low (a tight retest), and (optional) the candle rejects
       the level: it closes in its upper half. One signal per support (the same old low is never reused) plus a cooldown.
    No look-ahead. Columns: DCX_len, DCX_past, DCX_level, DCX_prev, DCX_cur, DCX_buy."""
    low = df["Low"].to_numpy(float); high = df["High"].to_numpy(float)
    close = df["Close"].to_numpy(float); op = df["Open"].to_numpy(float)
    n = len(low)
    atr = df["ATR"].to_numpy(float) if "ATR" in df else compute_atr(df, 14).to_numpy(float)
    ln, past, lvl_o = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    prv, cur_o, buy = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    last_buy, used = -10 ** 9, set()
    pn = int(pivot_n)
    ema = pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy()
    for t in range(base_len, n):
        lvl = low[t - base_len + 1:t + 1].min()
        lo_b = max(pn, t - int(max_back))
        hi_b = t - max(int(min_dist), base_len + 1) + 1
        if hi_b <= lo_b:
            continue
        a = atr[t] if np.isfinite(atr[t]) else 0.0
        j = -1
        if plain:                                                          # the 3 original rules only: nearest earlier candle at/below the level
            _h = np.nonzero(low[lo_b:hi_b] <= lvl)[0]
            if _h.size:
                j = lo_b + int(_h[-1])
        for jj in (() if plain else (lo_b + np.nonzero(low[lo_b:hi_b] <= lvl)[0])[::-1]):        # nearest first
            if low[jj - pn:jj + pn + 1].min() < low[jj]:                       # must be a swing low
                continue
            if high[jj:t + 1].max() - low[jj] < rally_atr * a:                 # must have bounced
                continue
            if low[jj + 1:t + 1].min() < low[jj]:                              # support must still be intact
                continue
            j = int(jj); break
        if j < 0:
            continue
        N = t - j
        cur = low[t - N + 1:t + 1].min(); prev = low[t - N:t].min()
        ln[t], past[t], lvl_o[t], prv[t], cur_o[t] = N, j, lvl, prev, cur
        rng = high[t] - low[t]
        ok_rej = (not reject) or (rng > 0 and close[t] >= low[t] + 0.5 * rng)
        ok_conf = (not confirm) or (t > 0 and close[t] > op[t] and close[t] > close[t - 1])   # bullish confirmation: green candle closing above the previous close
        ok_trend = (not trend_guard) or (t >= 10 and ema[t] >= ema[t - 10])  # not buying into a falling 50-EMA
        if plain:
            if cur > prev and low[t] <= cur and N >= min_dist:
                buy[t] = True
            continue
        if (cur > prev and low[t] <= cur and N >= min_dist and 0 <= low[t] - low[j] <= tol_atr * a
                and ok_rej and ok_conf and ok_trend and j not in used and t - last_buy > cooldown):
            buy[t] = True; last_buy = t; used.add(j)
    return pd.DataFrame({"DCX_len": ln, "DCX_past": past, "DCX_level": lvl_o, "DCX_prev": prv, "DCX_cur": cur_o, "DCX_buy": buy},
                        index=df.index)


# ---------------------------------------------------------------- Donchian extension: two-step BUY / SELL (mirror of Donchian_Auto_Extension.pine)
def _dcx_find_old(low, high, t, lvl, a, lo_b, hi_b, pn, rally_atr, plain, lowside):
    """Nearest earlier candle (support when lowside, resistance otherwise) as in the Pine script. Returns bar index or -1."""
    if hi_b <= lo_b:
        return -1
    if lowside:
        cand = lo_b + np.nonzero(low[lo_b:hi_b] <= lvl)[0]
    else:
        cand = lo_b + np.nonzero(high[lo_b:hi_b] >= lvl)[0]
    if cand.size == 0:
        return -1
    if plain:
        return int(cand[-1])
    for jj in cand[::-1]:                                                      # nearest first
        if lowside:
            if low[jj - pn:jj + pn + 1].min() < low[jj]:
                continue
            if high[jj:t + 1].max() - low[jj] < rally_atr * a:
                continue
            if low[jj + 1:t + 1].min() < low[jj]:
                continue
        else:
            if high[jj - pn:jj + pn + 1].max() > high[jj]:
                continue
            if high[jj] - low[jj:t + 1].min() < rally_atr * a:
                continue
            if high[jj + 1:t + 1].max() > high[jj]:
                continue
        return int(jj)
    return -1


def donchian_extension_steps(df, base_len=20, min_dist=30, max_back=1500, tol_atr=1.5, cooldown=8, pivot_n=3, rally_atr=2.0,
                             reject=False, trend_guard=False, plain=False, rsi_os=30.0, rsi_ob=70.0, rsi_now=True, wait_bars=0, **_ignored):
    """Two-step BUY / SELL, the same state machine as the TradingView script.
    STEP 1 (touch): the 8 checks pass - old swing level found, channel stepped (old level rolled out), bar touches the channel edge,
       within tol_atr x ATR of the old level, rejection candle (optional), trend guard (optional), old level not already used, cooldown.
       It arms the setup (dropped only if price breaks the touch candle's low/high, or after wait_bars when > 0).
    STEP 2 (confirmation): RSI(14) <= rsi_os for BUY (>= rsi_ob for SELL), on the touch candle itself or any later candle while armed.
       A step 2 is only valid while its step 1 is valid: any BUY/SELL step 1 or 2 on the other side cancels the pending step 1.
    Columns: DCX_len/past/level/prev/cur (current-bar preview of the BUY side), DCX_s1_buy, DCX_buy (= BUY step 2), DCX_arm_past/DCX_arm_len
    (old low behind a BUY step 1/2), DCXS_len/past (SELL side), DCX_s1_sell, DCX_sell (= SELL step 2), DCXS_arm_past/DCXS_arm_len. No look-ahead."""
    low = df["Low"].to_numpy(float); high = df["High"].to_numpy(float); close = df["Close"].to_numpy(float)
    n = len(low)
    atr = df["ATR"].to_numpy(float) if "ATR" in df else compute_atr(df, 14).to_numpy(float)
    rsi = compute_rsi(df["Close"], 14).to_numpy(float)
    ema = pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy()
    pn = int(pivot_n)
    nan = lambda: np.full(n, np.nan)
    b_len, b_past, b_lvl, b_prev, b_cur, b_apast, b_alen = nan(), nan(), nan(), nan(), nan(), nan(), nan()
    s_len, s_past, s_apast, s_alen = nan(), nan(), nan(), nan()
    s1_b, s2_b, s1_s, s2_s = (np.zeros(n, bool) for _ in range(4))
    last_buy = last_sell = -10 ** 9
    mark_buy = mark_sell = -10 ** 9
    used_b, used_s = None, None                                      # lastBuyPiv / lastSellPiv (index of the last used old low/high)
    arm_b = None; arm_s = None                                       # [extreme, bar, old_idx, N, rsi_min/max]
    s1_on_b = s1_on_s = False                                        # display flags of the Pine script (step 1 waiting)
    cd = int(cooldown)
    for t in range(base_len, n):
        a = atr[t] if np.isfinite(atr[t]) else 0.0
        lo_b = max(pn, t - int(max_back)); hi_b = t - max(int(min_dist), base_len + 1) + 1
        lvl_l = low[t - base_len + 1:t + 1].min(); lvl_h = high[t - base_len + 1:t + 1].max()
        jl = _dcx_find_old(low, high, t, lvl_l, a, lo_b, hi_b, pn, rally_atr, plain, True)
        jh = _dcx_find_old(low, high, t, lvl_h, a, lo_b, hi_b, pn, rally_atr, plain, False)
        buy_raw = sell_raw = False
        NL = NH = 0
        if jl >= 0:
            NL = t - jl
            cur = low[t - NL + 1:t + 1].min(); prev = low[t - NL:t].min()
            b_len[t], b_past[t], b_lvl[t], b_prev[t], b_cur[t] = NL, jl, lvl_l, prev, cur
            rng = high[t] - low[t]
            rej = (not reject) or (rng > 0 and close[t] >= low[t] + 0.5 * rng)
            trend = (not trend_guard) or plain or (t >= 10 and ema[t] >= ema[t - 10])
            fresh = plain or (jl != used_b)
            tol = plain or (0 <= low[t] - low[jl] <= tol_atr * a)
            buy_raw = cur > prev and low[t] <= cur and NL >= min_dist and tol and (plain or (rej and trend and fresh))
        if jh >= 0:
            NH = t - jh
            cur = high[t - NH + 1:t + 1].max(); prev = high[t - NH:t].max()
            s_len[t], s_past[t] = NH, jh
            rng = high[t] - low[t]
            rej = (not reject) or (rng > 0 and close[t] <= high[t] - 0.5 * rng)
            trend = (not trend_guard) or plain or (t >= 10 and ema[t] <= ema[t - 10])
            fresh = plain or (jh != used_s)
            tol = plain or (0 <= high[jh] - high[t] <= tol_atr * a)
            sell_raw = cur < prev and high[t] >= cur and NH >= min_dist and tol and (plain or (rej and trend and fresh))
        # ---------------- BUY side
        cd_buy = t - last_buy > cd
        buy_touch = buy_raw and cd_buy
        if buy_touch:
            arm_b = [low[t], t, jl, NL, np.nanmin(rsi[max(0, t - 4):t + 1]) if np.isfinite(rsi[max(0, t - 4):t + 1]).any() else np.nan]
            s1_on_b = True
        if arm_b is not None and t > arm_b[1]:
            arm_b[4] = np.nanmin([arm_b[4], rsi[t]]) if np.isfinite(rsi[t]) else arm_b[4]
            if low[t] < arm_b[0] or (wait_bars > 0 and t - arm_b[1] > wait_bars):
                arm_b = None
        if arm_b is not None:
            gate = (np.isfinite(rsi[t]) and rsi[t] <= rsi_os) if rsi_now else (np.isfinite(arm_b[4]) and arm_b[4] <= rsi_os)
            buy_setup = gate and t - last_buy > cd and (buy_touch or arm_b[1] > mark_sell)
        else:
            buy_setup = False
        show2_b = buy_setup and s1_on_b
        if buy_setup:
            last_buy = t; used_b = arm_b[2]
        buy_l1 = buy_raw and t - last_buy > cd
        if buy_touch:                                                  # touch candle: all 8 checks (incl. cooldown) were OK when it was judged
            s1_b[t] = True; b_apast[t] = jl; b_alen[t] = NL
        if show2_b:
            s2_b[t] = True; b_apast[t] = arm_b[2]; b_alen[t] = arm_b[3]
            s1_on_b = False
        if buy_l1 or buy_setup:
            mark_buy = t
        if buy_setup:
            arm_b = None
        if buy_l1 or buy_setup:
            arm_s = None                                               # a BUY signal cancels any pending SELL step 1
            s1_on_s = False
        # ---------------- SELL side
        cd_sell = t - last_sell > cd
        sell_touch = sell_raw and cd_sell
        if sell_touch:
            arm_s = [high[t], t, jh, NH, np.nanmax(rsi[max(0, t - 4):t + 1]) if np.isfinite(rsi[max(0, t - 4):t + 1]).any() else np.nan]
            s1_on_s = True
        if arm_s is not None and t > arm_s[1]:
            arm_s[4] = np.nanmax([arm_s[4], rsi[t]]) if np.isfinite(rsi[t]) else arm_s[4]
            if high[t] > arm_s[0] or (wait_bars > 0 and t - arm_s[1] > wait_bars):
                arm_s = None
        if arm_s is not None:
            gate = (np.isfinite(rsi[t]) and rsi[t] >= rsi_ob) if rsi_now else (np.isfinite(arm_s[4]) and arm_s[4] >= rsi_ob)
            sell_setup = gate and t - last_sell > cd and (sell_touch or arm_s[1] > mark_buy)
        else:
            sell_setup = False
        show2_s = sell_setup and s1_on_s
        if sell_setup:
            last_sell = t; used_s = arm_s[2]
        sell_l1 = sell_raw and t - last_sell > cd
        if sell_touch:
            s1_s[t] = True; s_apast[t] = jh; s_alen[t] = NH
        if show2_s:
            s2_s[t] = True; s_apast[t] = arm_s[2]; s_alen[t] = arm_s[3]
            s1_on_s = False
        if sell_l1 or sell_setup:
            mark_sell = t
            arm_b = None                                               # a SELL signal cancels any pending BUY step 1 (no stale BUY step 2)
            s1_on_b = False
        if sell_setup:
            arm_s = None
    return pd.DataFrame({"DCX_len": b_len, "DCX_past": b_past, "DCX_level": b_lvl, "DCX_prev": b_prev, "DCX_cur": b_cur,
                         "DCX_s1_buy": s1_b, "DCX_buy": s2_b, "DCX_arm_past": b_apast, "DCX_arm_len": b_alen,
                         "DCXS_len": s_len, "DCXS_past": s_past, "DCX_s1_sell": s1_s, "DCX_sell": s2_s,
                         "DCXS_arm_past": s_apast, "DCXS_arm_len": s_alen}, index=df.index)
