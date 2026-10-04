"""Local paper-trading engine. Fills at the CURRENT market price (not the forecast) with slippage."""
import json
import os
import sys
import datetime

from prepare_data import APP_DIR
from generate_forecast import get_live_prediction_detail

PORTFOLIO_FILE = os.path.join(APP_DIR, "local_sandbox_portfolio.json")
INITIAL_CASH = 100_000.00
POSITION_PCT = 0.10            # fraction of available cash committed per new position
MIN_CONFIDENCE = 0.55
REQUIRE_VALIDATED_EDGE = True  # refuse to trade a model that did not beat the baseline on validation data
SLIPPAGE = 0.0005              # 5 bps


def load_local_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        state = {"cash": INITIAL_CASH, "holdings": {}, "avg_cost": {}, "trades": []}
        save_local_portfolio(state)
        return state
    with open(PORTFOLIO_FILE) as f:
        state = json.load(f)
    changed = False
    if "shares_held" in state:                      # legacy single-ticker file
        state = {"cash": state["cash"], "holdings": {"MMM": state["shares_held"]}}
        changed = True
    for key, default in (("holdings", {}), ("avg_cost", {}), ("trades", [])):
        if key not in state:
            state[key], changed = default, True
    if changed:
        save_local_portfolio(state)
    return state


def save_local_portfolio(state):
    tmp = PORTFOLIO_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, PORTFOLIO_FILE)                 # atomic: never leaves a half-written file


def run_automated_trading_cycle(ticker):
    ticker = ticker.strip().upper()
    print(f"\n--- Paper trading cycle for {ticker} ---")
    pf = load_local_portfolio()
    held = pf["holdings"].get(ticker, 0)
    print(f"Cash: ${pf['cash']:,.2f} | holding {held} shares of {ticker}")

    try:
        d = get_live_prediction_detail(ticker)
    except Exception as e:
        print(f"Execution error: could not generate a forecast for {ticker}: {e}")
        return
    price, direction, conf = d["last_close"], d["direction"], d["confidence"]
    print(f"Signal: {direction} (P(up)={d['prob_up']:.1%}) | last close ${price:.2f} as of {d['as_of']}")
    if d["stale"]:
        print("Execution halted: market data is more than 5 days old.")
        return
    if conf < MIN_CONFIDENCE:
        print(f"Execution halted: confidence {conf:.1%} is below {MIN_CONFIDENCE:.0%}.")
        return
    if REQUIRE_VALIDATED_EDGE and not d["has_edge"]:
        m = d["meta"]
        print(f"Execution halted: this model did not beat the baseline on validation data "
              f"({m['val_accuracy']:.1%} vs {m['val_baseline']:.1%}).")
        return

    now = datetime.datetime.now().isoformat(timespec="seconds")
    if direction == "UP TREND" and held == 0:
        fill = price * (1 + SLIPPAGE)
        shares = int((pf["cash"] * POSITION_PCT) // fill)
        if shares < 1:
            print("Execution aborted: insufficient cash for one share.")
            return
        pf["cash"] -= shares * fill
        pf["holdings"][ticker] = shares
        pf["avg_cost"][ticker] = fill
        pf["trades"].append({"time": now, "ticker": ticker, "side": "BUY", "shares": shares, "price": fill})
        save_local_portfolio(pf)
        print(f"BUY executed: {shares} {ticker} @ ${fill:.2f} | cash ${pf['cash']:,.2f}")
    elif direction == "DOWN TREND" and held > 0:
        fill = price * (1 - SLIPPAGE)
        pf["cash"] += held * fill
        cost = pf["avg_cost"].get(ticker)
        pnl = (fill - cost) * held if cost else None
        pf["holdings"][ticker] = 0
        pf["avg_cost"].pop(ticker, None)
        pf["trades"].append({"time": now, "ticker": ticker, "side": "SELL", "shares": held, "price": fill, "pnl": pnl})
        save_local_portfolio(pf)
        print(f"SELL executed: {held} {ticker} @ ${fill:.2f}" + (f" | realised P&L ${pnl:+,.2f}" if pnl is not None else ""))
    else:
        print("No trade: stance already matches the portfolio.")


if __name__ == "__main__":
    run_automated_trading_cycle(sys.argv[1] if len(sys.argv) > 1 else "CRM")
