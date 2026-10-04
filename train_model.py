"""Train / retrain per-ticker models:  python train_model.py            (default watchlist)
                                       python train_model.py CRM NVDA     (specific tickers)"""
import sys
from generate_forecast import train_specific_ticker

DEFAULT = ["CRM", "MSFT", "NOW", "ORCL", "PLTR"]

if __name__ == "__main__":
    for t in (sys.argv[1:] or DEFAULT):
        try:
            m = train_specific_ticker(t)
            print(f"  -> saved. has_edge={m['has_edge']}  data through {m['data_through']}\n")
        except Exception as e:
            print(f"  !! {t} failed: {e}\n")
