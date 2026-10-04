"""Model training + live prediction (single-output LSTM: P(price higher in LEAD trading days))."""
import datetime
import json
import os

import joblib
import numpy as np

from prepare_data import (APP_DIR, FEATURE_COLUMNS, LEAD, LOOKBACK, build_dataset,
                          fit_scaler, scale_windows)

MIN_EDGE = 0.02          # model must beat the "always guess the majority class" baseline by this much
_CACHE = {}


class ModelNotReady(Exception):
    pass


def _paths(ticker):
    base = os.path.join(APP_DIR, f"vp_model_{ticker}")
    return base + ".keras", base + "_scaler.joblib", base + "_meta.json"


def build_model(timesteps, n_features):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(timesteps, n_features), name="Market_Data_Input")
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="Neural_Index_Output")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def fit_model(X_tr, y_tr, X_va, y_va, epochs=60, seed=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = build_model(X_tr.shape[1], X_tr.shape[2])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=32,
              callbacks=[stop], verbose=0)
    return model


def chrono_split(n, lead, val_frac=0.2):
    """Chronological split with an embargo of `lead` bars so no training label overlaps the validation period."""
    split = int(n * (1 - val_frac))
    return np.arange(0, max(split - lead, 0)), np.arange(split, n)


def train_specific_ticker(ticker):
    ticker = ticker.strip().upper()
    print(f"--- Training LSTM for {ticker} ---")
    ds = build_dataset(ticker)
    m = ~np.isnan(ds.y)
    Xl, yl = ds.X[m], ds.y[m]
    tr, va = chrono_split(len(Xl), ds.lead)
    if len(tr) < 300 or len(va) < 80:
        raise ValueError(f"Not enough labelled history to train {ticker} (train={len(tr)}, val={len(va)}).")
    scaler = fit_scaler(Xl[tr])
    Xtr, Xva = scale_windows(Xl[tr], scaler), scale_windows(Xl[va], scaler)
    model = fit_model(Xtr, yl[tr], Xva, yl[va])

    p = model.predict(Xva, verbose=0).ravel()
    val_acc = float(((p > 0.5) == (yl[va] > 0.5)).mean())
    up_rate = float(yl[va].mean())
    baseline = max(up_rate, 1 - up_rate)
    meta = {
        "ticker": ticker, "feature_names": FEATURE_COLUMNS, "lookback": ds.lookback, "lead": ds.lead,
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "val_accuracy": round(val_acc, 4), "val_baseline": round(baseline, 4),
        "val_edge": round(val_acc - baseline, 4), "has_edge": bool(val_acc - baseline >= MIN_EDGE),
        "val_prob_std": round(float(p.std()), 4),
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "data_through": str(ds.dates[-1].date()),
    }
    mp, sp, jp = _paths(ticker)
    model.save(mp)
    joblib.dump(scaler, sp)
    with open(jp, "w") as f:
        json.dump(meta, f, indent=2)
    _CACHE.pop(ticker, None)
    print(f"{ticker}: validation accuracy {val_acc:.1%} vs baseline {baseline:.1%} "
          f"(edge {val_acc - baseline:+.1%}, n={len(va)})")
    return meta


def load_bundle(ticker, auto_train=True):
    ticker = ticker.strip().upper()
    if ticker in _CACHE:
        return _CACHE[ticker]
    mp, sp, jp = _paths(ticker)
    compatible = False
    if all(os.path.exists(p) for p in (mp, sp, jp)):
        with open(jp) as f:
            meta = json.load(f)
        compatible = (meta.get("feature_names") == FEATURE_COLUMNS
                      and meta.get("lookback") == LOOKBACK and meta.get("lead") == LEAD)
    if not compatible:
        if not auto_train:
            raise ModelNotReady(f"No up-to-date model for {ticker}. Run: python train_model.py {ticker}")
        train_specific_ticker(ticker)
    import tensorflow as tf
    with open(jp) as f:
        meta = json.load(f)
    bundle = (tf.keras.models.load_model(mp), joblib.load(sp), meta)
    _CACHE[ticker] = bundle
    return bundle


def get_live_prediction_detail(ticker, auto_train=True):
    ticker = ticker.strip().upper()
    model, scaler, meta = load_bundle(ticker, auto_train)
    ds = build_dataset(ticker)
    p = float(model.predict(scale_windows(ds.X[-1:], scaler), verbose=0)[0, 0])   # newest bar, label not needed
    direction = "UP TREND" if p >= 0.5 else "DOWN TREND"
    last_close = float(ds.frame["Close"].iloc[-1])
    sigma = float(ds.frame["vol_14"].iloc[-1])
    h = np.sqrt(ds.lead)
    as_of = ds.dates[-1].date()
    return {
        "ticker": ticker, "as_of": str(as_of), "stale": (datetime.date.today() - as_of).days > 5,
        "last_close": last_close, "prob_up": p, "direction": direction, "confidence": max(p, 1 - p),
        # probability-weighted expected move, scaled by recent volatility (an estimate, not a model output)
        "expected_price": last_close * (1 + (2 * p - 1) * sigma * h),
        "band_low": last_close * (1 - sigma * h), "band_high": last_close * (1 + sigma * h),
        "daily_vol": sigma, "lead": ds.lead, "meta": meta, "has_edge": meta["has_edge"],
    }


def get_live_prediction(ticker="CRM"):
    """Backward-compatible tuple: (expected_price, direction, confidence)."""
    d = get_live_prediction_detail(ticker)
    return d["expected_price"], d["direction"], d["confidence"]


if __name__ == "__main__":
    import sys
    d = get_live_prediction_detail(sys.argv[1] if len(sys.argv) > 1 else "CRM")
    print(f"{d['ticker']} as of {d['as_of']}: P(up in {d['lead']}d)={d['prob_up']:.1%} -> {d['direction']} | "
          f"close ${d['last_close']:.2f}, expected ${d['expected_price']:.2f} "
          f"(1-sigma ${d['band_low']:.2f}-${d['band_high']:.2f}) | validated edge: {d['has_edge']}")
