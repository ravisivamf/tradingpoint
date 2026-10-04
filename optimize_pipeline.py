"""Grid-searches the structural reversal rule used by the dashboard, with in-sample / out-of-sample validation.
Never writes fabricated statistics: if nothing qualifies, the config says so."""
import json
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import compute_atr, compute_rsi
from prepare_data import APP_DIR

warnings.simplefilter("ignore", FutureWarning)
CONFIG_PATH = os.path.join(APP_DIR, "vp_config.json")
HORIZON, STOP_PCT, MIN_TRADES = 15, 0.03, 8


def run_structural_backtest(df, sr_lookback=60, rsi_buy=30, rsi_sell=70, start=0, end=None):
    """Non-overlapping reversal trades. Walls = prior `sr_lookback` bars (current bar excluded, same as the app).
    Signal on close of bar i -> entry at open of i+1. If stop and target are both touched in one bar the STOP is assumed
    to hit first. Time-outs are closed at the horizon close and counted (no hidden 'pending' trades)."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    rsi = compute_rsi(df["Close"]).values
    atr = compute_atr(df).values
    sup = pd.Series(l).rolling(sr_lookback).min().shift(1).values
    res = pd.Series(h).rolling(sr_lookback).max().shift(1).values
    end = len(df) - HORIZON - 2 if end is None else min(end, len(df) - HORIZON - 2)
    rets, i = [], max(start, sr_lookback + 15)
    while i < end:
        if np.isnan(sup[i]) or np.isnan(rsi[i]):
            i += 1
            continue
        long_ = c[i] <= sup[i] * 1.015 and rsi[i] <= rsi_buy
        short = c[i] >= res[i] * 0.985 and rsi[i] >= rsi_sell
        if not (long_ or short):
            i += 1
            continue
        entry = o[i + 1]
        side = 1 if long_ else -1
        tp, sl = (res[i], sup[i] - max(STOP_PCT * sup[i], 1.5 * atr[i])) if long_ else (sup[i], res[i] + max(STOP_PCT * res[i], 1.5 * atr[i]))
        exit_px, exit_j = c[i + HORIZON], i + HORIZON
        for j in range(i + 1, i + 1 + HORIZON):
            hit_sl = l[j] <= sl if long_ else h[j] >= sl
            hit_tp = h[j] >= tp if long_ else l[j] <= tp
            if hit_sl:
                exit_px, exit_j = sl, j
                break
            if hit_tp:
                exit_px, exit_j = tp, j
                break
        rets.append(side * (exit_px / entry - 1))
        i = exit_j + 1
    return rets


def summarise(rets):
    if not rets:
        return {"trades": 0, "win_rate": None, "avg_return": None, "profit_factor": None}
    w = [r for r in rets if r > 0]
    lo = [-r for r in rets if r < 0]
    return {"trades": len(rets), "win_rate": len(w) / len(rets), "avg_return": float(np.mean(rets)),
            "profit_factor": (sum(w) / sum(lo)) if lo else None}


def start_self_correcting_optimizer(ticker="CRM"):
    ticker = ticker.strip().upper()
    print(f"Downloading 5-year daily history for {ticker}...")
    data = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna(subset=["Close"]).reset_index(drop=True)
    split = int(len(data) * 0.7)

    best = None
    for lb, rb, rs in product([20, 40, 60, 120], [25, 30, 35, 40], [60, 65, 70, 75]):
        s = summarise(run_structural_backtest(data, lb, rb, rs, 0, split))
        if s["trades"] >= MIN_TRADES and s["avg_return"] is not None:
            if best is None or s["avg_return"] > best[0]["avg_return"]:
                best = (s, lb, rb, rs)

    if best is None:
        cfg = {"optimized_ticker": ticker, "sr_lookback": 60, "rsi_buy_entry_floor": 30.0,
               "rsi_sell_entry_ceiling": 70.0, "validated": False,
               "note": f"No parameter set produced >= {MIN_TRADES} in-sample trades; defaults kept."}
        print("No qualifying parameter set found. Defaults kept - nothing was 'verified'.")
    else:
        ins, lb, rb, rs = best
        oos = summarise(run_structural_backtest(data, lb, rb, rs, split, None))
        ok = oos["trades"] >= 5 and oos["avg_return"] is not None and oos["avg_return"] > 0
        cfg = {"optimized_ticker": ticker, "sr_lookback": lb, "rsi_buy_entry_floor": float(rb),
               "rsi_sell_entry_ceiling": float(rs), "in_sample": ins, "out_of_sample": oos, "validated": bool(ok)}
        print(f"Best in-sample: lookback={lb}, RSI buy<={rb}, sell>={rs} | in-sample {ins}")
        print(f"Out-of-sample (unseen 30%): {oos}")
        print("VALIDATED" if ok else "NOT validated out-of-sample - treat these parameters with caution.")
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)


if __name__ == "__main__":
    start_self_correcting_optimizer()
