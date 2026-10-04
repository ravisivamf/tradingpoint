"""Data pipeline: download -> stationary features -> leak-free supervised windows."""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from indicators import compute_rsi


def app_dir():
    """Folder that holds models/config/portfolio (next to the .exe when frozen)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = app_dir()
LOOKBACK = 10
LEAD = 4

# Local equity benchmark per Yahoo exchange suffix (default: SPY). The feature names keep the "spy_" prefix for
# backward compatibility, but they now hold the stock's OWN market index.
BENCHMARKS = {
    "NS": "^NSEI", "BO": "^BSESN", "L": "^FTSE", "T": "^N225", "HK": "^HSI", "TO": "^GSPTSE", "V": "^GSPTSE",
    "AX": "^AXJO", "DE": "^GDAXI", "F": "^GDAXI", "PA": "^FCHI", "AS": "^AEX", "MI": "FTSEMIB.MI", "MC": "^IBEX",
    "SW": "^SSMI", "ST": "^OMXS30", "SS": "000001.SS", "SZ": "399001.SZ", "KS": "^KS11", "KQ": "^KS11", "TW": "^TWII",
    "SI": "^STI", "SA": "^BVSP", "MX": "^MXX", "JK": "^JKSE", "KL": "^KLSE", "BK": "^SET.BK", "NZ": "^NZ50",
}


def benchmark_for(ticker):
    t = ticker.strip().upper()
    return BENCHMARKS.get(t.rsplit(".", 1)[1], "SPY") if "." in t else "SPY"


FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_14", "vol_14", "sma10_dist", "sma50_dist", "rsi_14",
    "macd_hist_pct", "range_pct", "gap_pct", "vol_ratio",
    "gold_ret_5", "bonds_ret_5", "spy_ret_1", "spy_ret_5", "rel_strength_5", "gold_bond_chg_14",
]


def download_ohlcv(ticker, period="5y", interval="1d"):
    raw = yf.download(ticker, period=period, interval=interval, progress=False,
                      auto_adjust=True, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"No market data returned for '{ticker}'. Check the symbol and your connection.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw.index = idx.normalize()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Data for '{ticker}' is missing columns: {missing}")
    return raw[cols].dropna(subset=["Close"])


def build_feature_frame(ticker, period="5y"):
    tgt = download_ohlcv(ticker, period)

    def macro(symbol):
        s = download_ohlcv(symbol, period)["Close"]
        # macro series trade on different calendars: align to the stock's dates, forward-fill only (no look-ahead)
        return s.reindex(tgt.index.union(s.index)).ffill().reindex(tgt.index)

    gold, bonds = macro("GC=F"), macro("^TNX")
    try:
        spy = macro(benchmark_for(ticker))          # the stock's own market index
    except Exception:
        spy = macro("SPY")                           # index unavailable on Yahoo -> global fallback
    c, o, h, l, v = tgt["Close"], tgt["Open"], tgt["High"], tgt["Low"], tgt["Volume"]

    f = pd.DataFrame(index=tgt.index)
    f["ret_1"] = c.pct_change()
    f["ret_5"] = c.pct_change(5)
    f["ret_14"] = c.pct_change(14)
    f["vol_14"] = f["ret_1"].rolling(14).std()
    f["sma10_dist"] = c / c.rolling(10).mean() - 1
    f["sma50_dist"] = c / c.rolling(50).mean() - 1
    f["rsi_14"] = compute_rsi(c) / 100.0
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["range_pct"] = (h - l) / c
    f["gap_pct"] = o / c.shift(1) - 1
    f["vol_ratio"] = np.log((v + 1.0) / (v.rolling(20).mean() + 1.0)).clip(-3, 3)
    f["gold_ret_5"] = gold.pct_change(5)
    f["bonds_ret_5"] = bonds.pct_change(5)
    f["spy_ret_1"] = spy.pct_change()
    f["spy_ret_5"] = spy.pct_change(5)
    f["rel_strength_5"] = f["ret_5"] - f["spy_ret_5"]
    f["gold_bond_chg_14"] = (gold / bonds).pct_change(14)

    frame = pd.concat([tgt, f], axis=1).replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=FEATURE_COLUMNS)


@dataclass
class Dataset:
    ticker: str
    X: np.ndarray            # (N, lookback, F) UNSCALED windows
    y: np.ndarray            # (N,) 1/0, NaN where the future is not yet known
    future_ret: np.ndarray   # (N,) realised forward return, NaN where unknown
    end_idx: np.ndarray      # frame row of each window's last bar
    dates: pd.DatetimeIndex
    frame: pd.DataFrame
    lookback: int
    lead: int


def build_dataset(ticker, lookback=LOOKBACK, lead=LEAD, period="5y"):
    """Window n covers bars [e-lookback+1 .. e] and is labelled with the return from close[e] to
    close[e+lead]. The newest windows have no label yet but ARE kept, so live prediction uses today's data."""
    ticker = ticker.strip().upper()
    frame = build_feature_frame(ticker, period)
    n = len(frame)
    if n < lookback + 120:
        raise ValueError(f"Not enough usable history for {ticker}: {n} rows.")
    feats = frame[FEATURE_COLUMNS].values.astype("float64")
    close = frame["Close"].values
    fut = np.full(n, np.nan)
    fut[:-lead] = close[lead:] / close[:-lead] - 1.0
    ends = np.arange(lookback - 1, n)
    X = np.stack([feats[e - lookback + 1:e + 1] for e in ends])
    fr = fut[ends]
    y = np.where(np.isnan(fr), np.nan, (fr > 0).astype(float))
    return Dataset(ticker, X, y, fr, ends, frame.index[ends], frame, lookback, lead)


def fit_scaler(X_train):
    """Fit on TRAINING windows only; never on data the model is later tested on."""
    return StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))


def scale_windows(X, scaler):
    Z = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return np.clip(Z, -5, 5).astype("float32")
