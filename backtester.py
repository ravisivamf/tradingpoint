"""Honest out-of-sample walk-forward backtest.
 - The model never sees the bars it is tested on (expanding window, retrained per fold, embargoed labels).
 - Signal at the close of bar t  ->  order filled at the OPEN of bar t+1, with trading costs.
 - Buy & hold is measured over exactly the same out-of-sample window.
Usage: python backtester.py CRM [--folds 4] [--cost-bps 5]"""
import argparse

import numpy as np

from prepare_data import build_dataset, fit_scaler, scale_windows


def oos_probabilities(ds, n_folds=4, min_train_frac=0.5):
    from generate_forecast import fit_model
    n, lead = len(ds.X), ds.lead
    labelled = ~np.isnan(ds.y)
    edges = np.linspace(int(n * min_train_frac), n, n_folds + 1).astype(int)
    probs = np.full(n, np.nan)
    for k in range(n_folds):
        a, b = edges[k], edges[k + 1]
        pool = np.where(labelled & (np.arange(n) < a - lead))[0]       # embargo: labels end before the test block
        cut = int(len(pool) * 0.85)
        tr, va = pool[:max(cut - lead, 0)], pool[cut:]
        scaler = fit_scaler(ds.X[tr])
        model = fit_model(scale_windows(ds.X[tr], scaler), ds.y[tr],
                          scale_windows(ds.X[va], scaler), ds.y[va], epochs=40)
        probs[a:b] = model.predict(scale_windows(ds.X[a:b], scaler), verbose=0).ravel()
        print(f"  fold {k + 1}/{n_folds}: trained on {len(tr)} windows, tested on {b - a} bars")
    return probs, edges[0]


def simulate(opens, closes, probs, buy_th, sell_th, cost, capital, always_long=False):
    n = len(closes)
    cash, shares, entry, pending = capital, 0.0, None, ("buy" if always_long else None)
    equity, trades, held = np.empty(n), [], 0
    for i in range(n):
        if pending == "buy" and shares == 0:
            px = opens[i] * (1 + cost)
            shares, cash, entry = cash / px, 0.0, px
        elif pending == "sell" and shares > 0:
            px = opens[i] * (1 - cost)
            cash, shares = shares * px, 0.0
            trades.append(px / entry - 1)
        pending = None
        equity[i] = cash + shares * closes[i]
        held += shares > 0
        if not always_long:
            if probs[i] > buy_th and shares == 0:
                pending = "buy"
            elif probs[i] < sell_th and shares > 0:
                pending = "sell"
    if shares > 0:                                    # liquidate at the end so open trades are counted
        px = closes[-1] * (1 - cost)
        trades.append(px / entry - 1)
        equity[-1] = shares * px
    return equity, trades, held / n


def performance(equity, trades, exposure, capital):
    eq = np.r_[capital, equity]
    rets = np.diff(eq) / eq[:-1]
    years = len(equity) / 252
    wins = [t for t in trades if t > 0]
    losses = [-t for t in trades if t < 0]
    return {
        "total_return": equity[-1] / capital - 1,
        "cagr": (equity[-1] / capital) ** (1 / years) - 1 if years > 0 else float("nan"),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0,
        "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()),
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else float("nan"),
        "profit_factor": (sum(wins) / sum(losses)) if losses else (float("inf") if wins else float("nan")),
        "exposure": exposure,
        "final_equity": float(equity[-1]),
    }


def run_backtest(ticker, n_folds=4, buy_th=0.55, sell_th=0.45, cost_bps=5.0, capital=10_000.0):
    ticker = ticker.strip().upper()
    print(f"--- Walk-forward backtest: {ticker} ---")
    ds = build_dataset(ticker)
    probs, first = oos_probabilities(ds, n_folds)
    rows = ds.end_idx[first:]
    opens = ds.frame["Open"].values[rows]
    closes = ds.frame["Close"].values[rows]
    p = probs[first:]
    cost = cost_bps / 10_000
    lab = ~np.isnan(ds.y[first:])
    acc = float(((p[lab] > 0.5) == (ds.y[first:][lab] > 0.5)).mean())
    up = float(ds.y[first:][lab].mean())

    bh = performance(*simulate(opens[1:], closes[1:], None, 0, 0, cost, capital, True), capital)
    print(f"\nOut-of-sample window: {ds.dates[first].date()} -> {ds.dates[-1].date()} ({len(p)} bars)")
    print(f"Directional accuracy: {acc:.1%}  (always-guess-majority baseline: {max(up, 1 - up):.1%}, n={lab.sum()})")
    print("\nThreshold sensitivity (same predictions, different trade triggers):")
    print(f"{'buy>':>6}{'sell<':>7}{'return':>9}{'Sharpe':>8}{'maxDD':>8}{'trades':>8}{'win%':>7}{'PF':>7}{'expo':>7}")
    results = {}
    for b, s in sorted({(0.50, 0.50), (0.52, 0.48), (buy_th, sell_th)}):
        r = performance(*simulate(opens, closes, p, b, s, cost, capital), capital)
        results[(b, s)] = r
        print(f"{b:>6.2f}{s:>7.2f}{r['total_return']:>9.1%}{r['sharpe']:>8.2f}{r['max_drawdown']:>8.1%}"
              f"{r['trades']:>8}{r['win_rate']:>7.0%}{r['profit_factor']:>7.2f}{r['exposure']:>7.0%}")
    print(f"\nBuy & hold, same window: return {bh['total_return']:.1%}, Sharpe {bh['sharpe']:.2f}, maxDD {bh['max_drawdown']:.1%}")
    main = results[(buy_th, sell_th)]
    print(f"\nStrategy (buy>{buy_th}, sell<{sell_th}, {cost_bps:g} bps/side): ${capital:,.0f} -> ${main['final_equity']:,.2f}")
    if main["trades"] < 30:
        print(f"NOTE: only {main['trades']} trades - too few for these numbers to be statistically meaningful.")
    return {"oos": results, "buy_hold": bh, "accuracy": acc, "baseline": max(up, 1 - up)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="CRM")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--buy", type=float, default=0.55)
    ap.add_argument("--sell", type=float, default=0.45)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    a = ap.parse_args()
    run_backtest(a.ticker, a.folds, a.buy, a.sell, a.cost_bps)
