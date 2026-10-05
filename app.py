import datetime
import importlib.util
import json
import os
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Tradepoint", layout="wide")


def _password_gate():
    """Optional login for a public website: set the TRADEPOINT_PASSWORD environment variable (or Streamlit secret)."""
    import hmac
    pw = os.environ.get("TRADEPOINT_PASSWORD")
    if not pw:
        try:
            pw = st.secrets.get("TRADEPOINT_PASSWORD")
        except Exception:
            pw = None
    if not pw or st.session_state.get("_authed"):
        return
    st.markdown("## 🔒 Tradepoint")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), str(pw).encode()):
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


_password_gate()

from bos_signals import compute_rsi_divergence, compute_structure
from fetch_news import get_news_detail
from indicators import (auto_edge_donchian, compute_adx_engine, compute_atr, compute_edge_donchian, compute_institutional_pivots,
                        compute_rsi, compute_zero_lag_ema, get_auto_donchian_length)
from prepare_data import APP_DIR

PORTFOLIO_FILE = os.path.join(APP_DIR, "local_sandbox_portfolio.json")
CONFIG_FILE = os.path.join(APP_DIR, "vp_config.json")


# ------------------------------------------------------------------ data
@st.cache_data(ttl=60)
def high_speed_market_download(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def load_chart_data(ticker, cfg):
    df = high_speed_market_download(ticker, cfg["period"], cfg["interval"])
    if not df.empty and cfg.get("resample"):
        # real 2H/4H bars aligned to the 9:30 open (previously these were just 1H bars with a different label)
        firsts = pd.Series(df.index, index=df.index).groupby(df.index.date).first()
        session_open = (firsts - firsts.dt.normalize()).mode().iloc[0]       # e.g. 9:30 NYSE, 9:15 NSE, 9:00 Tokyo, 8:00 Xetra
        df = (df.resample(cfg["resample"], offset=session_open, origin="start_day")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna(subset=["Close"]))
    return df


CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹", "GBP": "£", "GBp": "p ", "EUR": "€", "JPY": "¥", "CNY": "¥", "HKD": "HK$",
                    "AUD": "A$", "CAD": "C$", "CHF": "CHF ", "KRW": "₩", "TWD": "NT$", "SGD": "S$", "BRL": "R$",
                    "MXN": "MX$", "SEK": "kr ", "NOK": "kr ", "DKK": "kr ", "ZAR": "R ", "NZD": "NZ$", "IDR": "Rp ",
                    "MYR": "RM ", "THB": "฿", "ILS": "₪", "ILA": "ag ", "TRY": "₺", "PLN": "zł "}


@st.cache_data(ttl=86400)
def get_currency_symbol(ticker):
    try:
        cur = yf.Ticker(ticker).fast_info["currency"]
        return CURRENCY_SYMBOLS.get(cur, f"{cur} ")
    except Exception:
        return "$" if "." not in ticker else ""


@st.cache_data(ttl=3600)
def search_symbols(query):
    try:
        return [(q.get("symbol", ""), q.get("shortname") or q.get("longname") or "", q.get("exchDisp") or "")
                for q in yf.Search(query, max_results=8).quotes]
    except Exception:
        return []


@st.cache_data(ttl=600)
def cached_news(ticker):
    return get_news_detail(ticker)


@st.cache_data(ttl=900)
def neural_snapshot(ticker):
    try:
        from generate_forecast import get_live_prediction_detail
        return get_live_prediction_detail(ticker, auto_train=False)
    except Exception as e:
        return {"error": str(e)}


def load_portfolio():
    try:
        with open(PORTFOLIO_FILE) as f:
            s = json.load(f)
        if "shares_held" in s:
            s = {"cash": s["cash"], "holdings": {"MMM": s["shares_held"]}}
        s.setdefault("holdings", {}); s.setdefault("avg_cost", {}); s.setdefault("trades", [])
        return s
    except Exception:
        return {"cash": 100000.0, "holdings": {}, "avg_cost": {}, "trades": []}


# ------------------------------------------------------------------ theme
st.sidebar.header("🎨 App Interface Controls")
theme_selection = st.sidebar.selectbox("Select Display Theme Layout", ["Industrial Dark", "Classic Light"])
if theme_selection == "Industrial Dark":
    bg_color, text_color, muted_color = "#14171D", "#F1F5F9", "#AAB4C3"
    card_bg, input_bg, border_color, grid_color, template = "#1F232C", "#2A2F3A", "#3A4150", "#2F3541", "plotly_dark"
    metric_value_color = "#FFFFFF"
else:
    bg_color, text_color, muted_color = "#F5F7FA", "#0F172A", "#475569"
    card_bg, input_bg, border_color, grid_color, template = "#FFFFFF", "#FFFFFF", "#CBD5E1", "#E2E8F0", "plotly_white"
    metric_value_color = "#0F172A"

st.markdown(f"""
<style>
/* ---- surfaces ---- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {{ background-color: {bg_color} !important; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{ background-color: {card_bg} !important; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {border_color}; width: 400px !important; }}
.stMainBlockContainer {{ max-width: 95% !important; padding-left: 2% !important; padding-right: 2% !important; }}
/* ---- every kind of text, in both themes ---- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *, [data-testid="stWidgetLabel"] *,
[data-testid="stCheckbox"] *, [data-testid="stRadio"] *, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
[data-testid="stToolbar"] *, [data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color: {text_color} !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {muted_color} !important; }}
html, body, [data-testid="stWidgetLabel"] p {{ font-size: 22px !important; font-weight: 500 !important; }}
code, [data-testid="stMarkdownContainer"] code {{ background-color: {input_bg} !important; color: #00C896 !important; border: 1px solid {border_color}; }}
/* ---- metrics ---- */
div[data-testid="metric-container"] {{ background-color: {card_bg} !important; border: 2px solid {border_color} !important; padding: 24px !important; border-radius: 8px !important; }}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{ font-size: 36px !important; font-weight: 800 !important; color: {metric_value_color} !important; }}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {{ font-size: 18px !important; font-weight: bold !important; text-transform: uppercase; color: {muted_color} !important; }}
/* ---- inputs, selects, buttons ---- */
input, textarea, [data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="base-input"] {{ background-color: {input_bg} !important; color: {text_color} !important; border-color: {border_color} !important; }}
input, textarea, [data-baseweb="select"] * {{ color: {text_color} !important; -webkit-text-fill-color: {text_color} !important; }}
.stTextInput>div>div>input {{ font-size: 20px !important; padding: 12px !important; }}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul, [data-baseweb="menu"] {{ background-color: {card_bg} !important; }}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {{ color: {text_color} !important; background-color: {card_bg} !important; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {{ background-color: {input_bg} !important; }}
.stButton button, [data-testid="stBaseButton-secondary"] {{ background-color: {input_bg} !important; color: {text_color} !important; border: 1px solid {border_color} !important; }}
.stButton button *, [data-testid="stBaseButton-secondary"] * {{ color: {text_color} !important; }}
/* ---- alert boxes: neutral card + coloured edge so text is always readable ---- */
[data-testid="stAlert"] {{ background-color: {card_bg} !important; border: 1px solid {border_color} !important; border-radius: 8px !important; }}
[data-testid="stAlert"], [data-testid="stAlert"] * {{ color: {text_color} !important; }}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {{ border-left: 8px solid #00B0FF !important; }}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{ border-left: 8px solid #00E676 !important; }}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {{ border-left: 8px solid #FFB300 !important; }}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{ border-left: 8px solid #FF5252 !important; }}
/* ---- expanders and tables ---- */
[data-testid="stExpander"] {{ background-color: {card_bg} !important; border: 1px solid {border_color} !important; border-radius: 8px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: {text_color} !important; }}
.stTable table, [data-testid="stTable"] table {{ font-size: 20px !important; width: 100% !important; background-color: {card_bg} !important; }}
.stTable th, .stTable td, [data-testid="stTable"] th, [data-testid="stTable"] td {{ color: {text_color} !important; background-color: {card_bg} !important; border-color: {border_color} !important; }}
.stTable th, [data-testid="stTable"] th {{ background-color: {input_bg} !important; }}
</style>""", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 42px;'>📊 Tradepoint - Macro Predictive Platform</h1>", unsafe_allow_html=True)
# page skeleton: headline numbers, then the replay bar and the chart together; the analysis panels follow below the chart
metrics_slot = st.container()
replay_slot = st.container()
chart_slot = st.container()

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Market Asset Settings")
selected_ticker = st.sidebar.text_input("Type Main Chart Ticker", value="CRM").strip().upper()
st.sidebar.caption("Any Yahoo Finance symbol: US `CRM` · India `RELIANCE.NS` / `TCS.BO` · UK `VOD.L` · Germany `SAP.DE` · "
                   "Japan `7203.T` · Hong Kong `0700.HK` · Canada `SHOP.TO` · Australia `BHP.AX` · crypto `BTC-USD` · indices `^GSPC`")
_q = st.sidebar.text_input("🔎 Don't know the symbol? Search by company name", value="")
if _q.strip():
    _hits = search_symbols(_q.strip())
    st.sidebar.caption("\n".join(f"`{a}` - {b} ({c})" for a, b, c in _hits) if _hits else "No matches.")
if not selected_ticker:
    st.error("Please enter a valid ticker symbol.")
    st.stop()

tf_config = {
    "Daily (1D)":      {"interval": "1d",  "period": "5y",  "is_intraday": False, "resample": None,  "bar_width": 0.4, "auto_bars": 15},
    "Weekly (1W)":     {"interval": "1wk", "period": "5y",  "is_intraday": False, "resample": None,  "bar_width": 0.4, "auto_bars": 12},
    "Monthly (1M)":    {"interval": "1mo", "period": "5y",  "is_intraday": False, "resample": None,  "bar_width": 0.4, "auto_bars": 6},
    "4-Hour (4H)":     {"interval": "60m", "period": "6mo", "is_intraday": True,  "resample": "4h",  "bar_width": 0.4, "auto_bars": 3},
    "2-Hour (2H)":     {"interval": "60m", "period": "6mo", "is_intraday": True,  "resample": "2h",  "bar_width": 0.4, "auto_bars": 4},
    "1-Hour (1H)":     {"interval": "1h",  "period": "3mo", "is_intraday": True,  "resample": None,  "bar_width": 0.4, "auto_bars": 7},
    "30-Minute (30M)": {"interval": "30m", "period": "1mo", "is_intraday": True,  "resample": None,  "bar_width": 0.4, "auto_bars": 14},
    "15-Minute (15M)": {"interval": "15m", "period": "1mo", "is_intraday": True,  "resample": None,  "bar_width": 0.4, "auto_bars": 15},
    "5-Minute (5M)":   {"interval": "5m",  "period": "5d",  "is_intraday": True,  "resample": None,  "bar_width": 0.4, "auto_bars": 15},
}
selected_tf = st.sidebar.selectbox("Select Chart Resolution Timeframe", list(tf_config))
st.sidebar.markdown("#### 🔭 Chart View")
show_bars = st.sidebar.slider("Bars shown on chart", 40, 500, 150, 10, help="Fewer bars = bigger candles. All indicators are still calculated on the full history.")
fit_y = st.sidebar.checkbox("Fit price axis to candles", value=True, help="Keeps candles full-size; far-away levels (stop, target) no longer squash the chart.")
crosshair = st.sidebar.checkbox("Crosshair cursor lines", value=True, help="Dotted lines that follow the mouse across all panels, with price/time read-outs on the axes.")
chart_drag = st.sidebar.selectbox("Mouse drag does", ["Pan", "Zoom box"], help="Mouse wheel always zooms. Double-click the chart to reset.")
improved_rules = st.sidebar.checkbox("Improved entry & target rules", value=True,
    help="Target = nearest of wall / Donchian edge / ATR reach for the forecast horizon (not always the far wall). Stop = 1.5 ATR beyond the wall (not 3%). "
         "A fresh RSI divergence near a wall (within 3%) also counts as a reversal entry. Untick to use the old rules; the hit-rate section compares both.")
active_cfg = tf_config[selected_tf]

df_raw = load_chart_data(selected_ticker, active_cfg)
if df_raw.empty:
    st.sidebar.warning("⚠️ Intraday data unavailable - falling back to Daily.")
    active_cfg = tf_config["Daily (1D)"]
    df_raw = load_chart_data(selected_ticker, active_cfg)
if df_raw.empty:
    st.error(f"No market data found for '{selected_ticker}'. Non-US stocks need the Yahoo exchange suffix (e.g. RELIANCE.NS, VOD.L, 7203.T).")
    _sug = search_symbols(selected_ticker)
    if _sug:
        st.info("Did you mean: " + ", ".join(f"`{a}` ({b}, {c})" for a, b, c in _sug[:6]))
    st.stop()

cs = get_currency_symbol(selected_ticker)
csm = cs.replace("$", "\\$")      # markdown-safe version (a bare $ starts LaTeX in Streamlit)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Indicator Layers")
toggle_bb = st.sidebar.checkbox("Enable Bollinger Bands", value=True)
bb_color = st.sidebar.color_picker("Bollinger Band color", "#FFA726")
bb_width = st.sidebar.slider("Bollinger Band thickness", 1, 4, 2)
toggle_zlema = st.sidebar.checkbox("Enable Zero-Lag Trend Signal", value=True)
toggle_pivots = st.sidebar.checkbox("Enable Institutional Pivots (last 20 bars)", value=False)
pivot_mode = st.sidebar.selectbox("Pivot Computation Style", ["Traditional", "Fibonacci"])
st.sidebar.markdown("#### Donchian Channel")
auto_mode_dc = st.sidebar.checkbox("Auto Set Length By Active Ticker", value=True)
toggle_dc = st.sidebar.checkbox("Show Donchian Channel", value=False)
dc_style = st.sidebar.selectbox("Channel style", ["Edge candles (pullback pivots)", "Classic (every candle)"])
if auto_mode_dc:
    length_DC, dc_n = auto_edge_donchian(df_raw, get_auto_donchian_length(selected_ticker))
    st.sidebar.info(f"🎯 Auto: **{length_DC} bars**, edge swing **{dc_n}** - measured from {selected_ticker}'s own swing spacing on {selected_tf}")
else:
    length_DC = st.sidebar.slider("Manual Donchian Channel Lookback", 2, 100, 20)
    dc_n = st.sidebar.slider("Edge swing strength (bars each side)", 2, 8, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧱 Structure Marks (BOS / CHoCH) - chart only")
show_bos = st.sidebar.checkbox("Show BOS / CHoCH marks", value=True)
bos_n = st.sidebar.slider("Swing strength (bars each side)", 2, 10, 5, help="Bigger = fewer, more significant structure points.")
bos_disp = st.sidebar.slider("Break strength (x ATR beyond the level)", 0.1, 1.5, 0.3, 0.05)
bos_swing = st.sidebar.slider("Minimum swing size (x ATR)", 0.5, 4.0, 1.0, 0.25)
st.sidebar.caption("Marks are informational only: they do NOT change the stance, target, stop, probability or any statistics.")
st.sidebar.markdown("#### RSI-Divergence at Support / Resistance")
show_div_sr = st.sidebar.checkbox("Show RSI-divergence entries & exits", value=True)
swing_n = st.sidebar.slider("Divergence swing strength (bars each side)", 2, 8, 3)
div_tol = st.sidebar.slider("'At the line' tolerance (%)", 0.5, 3.0, 1.5, 0.25)
div_window = st.sidebar.slider("Divergence stays armed (bars)", 3, 30, 10, help="After a divergence forms away from the line, a later touch of support/resistance inside this window triggers the signal.")
bos_rr = st.sidebar.slider("Target = R multiple of risk", 1.0, 4.0, 2.0, 0.5)
long_only = st.sidebar.checkbox("Long only (divergence trades)", value=False)
st.sidebar.caption("Tuning these sliders to the table is curve-fitting - judge by the out-of-sample row.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Bar Replay / Past Testing")
st.sidebar.caption("The replay bar now sits directly above the chart.")
max_replay = max(0, min(500, len(df_raw) - 130))
replay_bars_back = 0
reveal_future = False
_REPLAY_SPEEDS = {"10x": 0.05, "7x": 0.2, "3x": 0.5, "1x": 1.0, "0.3x": 3.0, "0.1x": 10.0}   # pause between bars (the app's own compute time adds to it)
with replay_slot:
    replay_mode = st.checkbox("⏱️ Bar Replay / Past Testing", value=False, key="tp_replay_on",
                              help="Replays the market bar by bar from a past point. The system only sees data up to the replay bar.")
    if replay_mode and max_replay == 0:
        st.warning("Not enough history for replay on this timeframe.")
    elif replay_mode:
        _rk = f"{selected_ticker}|{selected_tf}"
        if st.session_state.get("tp_rk") != _rk:
            st.session_state.update(tp_rk=_rk, tp_rb=min(100, max_replay), tp_play=False)
        if "tp_pending" in st.session_state:
            st.session_state["tp_rb"] = st.session_state.pop("tp_pending")
        st.session_state["tp_max"] = max_replay
        st.session_state["tp_idx"] = [t.tz_localize(None) if t.tzinfo else t for t in pd.DatetimeIndex(df_raw.index)]
        st.session_state["tp_rb"] = int(min(max(st.session_state.get("tp_rb", 0), 0), max_replay))

        def _cb_step(d):
            st.session_state["tp_rb"] = int(min(max(st.session_state["tp_rb"] + d, 0), st.session_state["tp_max"]))
            st.session_state["tp_play"] = False
        def _cb_play():
            if st.session_state["tp_rb"] > 0:
                st.session_state["tp_play"] = not st.session_state.get("tp_play", False)
        def _cb_live():
            st.session_state.update(tp_rb=0, tp_play=False)
        def _cb_select():
            idx = st.session_state["tp_idx"]
            tgt = pd.Timestamp(st.session_state["tp_date"])
            if st.session_state.get("tp_time") is not None and active_cfg["is_intraday"]:
                tgt = tgt.normalize() + pd.Timedelta(hours=st.session_state["tp_time"].hour, minutes=st.session_state["tp_time"].minute)
            else:
                tgt = tgt.normalize() + pd.Timedelta(hours=23, minutes=59)
            pos = int(pd.DatetimeIndex(idx).searchsorted(tgt, side="right"))     # bars at or before the chosen moment
            st.session_state["tp_rb"] = int(min(max(len(idx) - pos, 0), st.session_state["tp_max"]))
            st.session_state["tp_play"] = False

        _n = len(st.session_state["tp_idx"])
        _first = st.session_state["tp_idx"][max(_n - 1 - max_replay, 0)]
        _last = st.session_state["tp_idx"][-1]
        _cur = st.session_state["tp_idx"][_n - 1 - st.session_state["tp_rb"]]
        _playing = bool(st.session_state.get("tp_play", False)) and st.session_state["tp_rb"] > 0
        _intr = active_cfg["is_intraday"]
        _r1 = st.columns([1.5] + ([1.2] if _intr else []) + [1.4, 0.5, 0.5, 0.5, 0.5], vertical_alignment="bottom")
        _k = 0
        _r1[_k].date_input("Select bar - start date", value=_cur.date(), min_value=_first.date(), max_value=_last.date(), key="tp_date"); _k += 1
        if _intr:
            _r1[_k].time_input("Start time", value=_cur.time(), key="tp_time", step=300); _k += 1
        _r1[_k].button("📍 Go to bar", on_click=_cb_select, use_container_width=True); _k += 1
        _r1[_k].button("⏮", on_click=_cb_step, args=(1,), help="Step back one bar", use_container_width=True); _k += 1
        _r1[_k].button("⏸" if _playing else "▶", on_click=_cb_play, help="Play / pause", use_container_width=True); _k += 1
        _r1[_k].button("⏭", on_click=_cb_step, args=(-1,), help="Step forward one bar", use_container_width=True); _k += 1
        _r1[_k].button("⏩", on_click=_cb_live, help="Jump to real-time", use_container_width=True)
        _r2 = st.columns([3.2, 1.0, 1.8], vertical_alignment="bottom")
        replay_bars_back = _r2[0].slider("Replay position (bars before latest) - drag to scrub, 0 = real-time", 0, max_replay, key="tp_rb")
        replay_speed = _r2[1].selectbox("Speed", list(_REPLAY_SPEEDS), index=3,
                                        help="10x fastest, 0.1x slowest. The app recalculates every bar, so very fast speeds are limited by the server.")
        reveal_future = _r2[2].checkbox("Reveal what happens next", value=False,
                                        help="Off = blind replay like TradingView: only bars up to the replay point are drawn. On = faded future bars + scorecard.")
        if replay_bars_back == 0:
            st.success("⚡ REAL-TIME: anchored to the latest close.")
        else:
            _at = st.session_state["tp_idx"][_n - 1 - replay_bars_back]
            st.warning(f"⏮️ REPLAY at **{_at:%Y-%m-%d %H:%M}** - {replay_bars_back} bars before the latest" + (" - ▶ playing" if _playing else ""))
        st.session_state["tp_playing_now"] = _playing
        st.session_state["tp_delay"] = _REPLAY_SPEEDS[replay_speed]

st.sidebar.markdown("### Macro Forecasting Scope")
scope_mode = st.sidebar.radio("Horizon Selection Mode", ["Auto-Pilot", "Manual Control"])
if scope_mode == "Auto-Pilot":
    forecast_lead_units = int(active_cfg["auto_bars"])
    st.sidebar.info(f"🤖 Auto-Pilot Active: Locked to **{forecast_lead_units} Units**")
else:
    forecast_lead_units = st.sidebar.slider(
        "Intraday Horizon (Future Bars)" if active_cfg["is_intraday"] else "Macro Horizon (Bars Out)",
        1, 50 if active_cfg["is_intraday"] else 250, int(active_cfg["auto_bars"]))

st.sidebar.markdown("---")
st.sidebar.markdown("### Scanner Watchlist Configuration")
watchlist_raw = st.sidebar.text_input("Edit Watchlist Symbols (Comma Separated)", value="MSFT, NOW, ORCL, PLTR, CRM, VST")
watchlist = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]

# ------------------------------------------------------------------ config from optimizer
rsi_buy_floor, rsi_sell_ceil, sr_lookback, cfg_note = 30.0, 70.0, 60, None
try:
    with open(CONFIG_FILE) as f:
        saved = json.load(f)
    rsi_buy_floor = float(saved.get("rsi_buy_entry_floor", rsi_buy_floor))
    rsi_sell_ceil = float(saved.get("rsi_sell_entry_ceiling", rsi_sell_ceil))
    sr_lookback = max(20, int(saved.get("sr_lookback", sr_lookback)))   # legacy "structural_pivot_lookback" (5) is ignored: it was never a real optimised value
    if saved.get("optimized_ticker") != selected_ticker:
        cfg_note = f"Optimizer parameters were tuned on {saved.get('optimized_ticker')}, not {selected_ticker}."
    elif not saved.get("validated", False):
        cfg_note = "Optimizer parameters did not validate out-of-sample."
except Exception:
    pass
if cfg_note:
    st.sidebar.caption(f"⚠️ {cfg_note}")

sr_eff = min(sr_lookback, max(10, len(df_raw) // 4))   # short histories (e.g. monthly) cannot support a long wall window
# ------------------------------------------------------------------ indicators on FULL history, then slice (no warm-up distortion, no look-ahead)
all_df = df_raw.copy()
all_df["Short_MA"] = all_df["Close"].rolling(10).mean()
all_df["Long_MA"] = all_df["Close"].rolling(30).mean()
all_df["Long_Diff"] = (all_df["Close"].rolling(10).mean() - all_df["Close"].rolling(50).mean()).rolling(5).mean()
all_df["Medium_Diff"] = (all_df["Close"] - all_df["Close"].rolling(20).mean()).rolling(5).mean()
all_df["Short_Diff"] = (all_df["Close"] - all_df["Open"]).rolling(5).mean()
all_df["BB_Basis"] = all_df["Close"].rolling(20).mean()
bb_std = all_df["Close"].rolling(20).std()
all_df["BB_Upper"], all_df["BB_Lower"] = all_df["BB_Basis"] + 2 * bb_std, all_df["BB_Basis"] - 2 * bb_std
all_df["ZLEMA"] = compute_zero_lag_ema(all_df["Close"], 70)
all_df["ADX"] = compute_adx_engine(all_df, 14)
all_df["ATR"] = compute_atr(all_df, 14)
all_df["RSI_14"] = compute_rsi(all_df["Close"])
if dc_style.startswith("Edge"):
    _dc = compute_edge_donchian(all_df, length_DC, dc_n)
    for _c in _dc.columns:
        all_df[_c] = _dc[_c]
else:
    all_df["DC_Upper"] = all_df["High"].rolling(length_DC).max()
    all_df["DC_Lower"] = all_df["Low"].rolling(length_DC).min()
    all_df["DC_up_bar"] = np.nan
    all_df["DC_lo_bar"] = np.nan
# structural walls: PRIOR N bars, current bar excluded (identical to optimize_pipeline.py)
all_df["Wall_Res"] = all_df["High"].rolling(sr_eff).max().shift(1)
all_df["Wall_Sup"] = all_df["Low"].rolling(sr_eff).min().shift(1)

_st = compute_structure(all_df, n=bos_n, disp_atr=bos_disp, min_swing_atr=bos_swing)
for _c in _st.columns:
    all_df["BOS_" + _c] = _st[_c]

_dv = compute_rsi_divergence(all_df, n=swing_n)
for _c in _dv.columns:
    all_df["DIV_" + _c] = _dv[_c]
# confluence: +1 bullish / -1 bearish for each independent evidence group
_ok = all_df["Wall_Sup"].notna() & all_df["RSI_14"].notna()
all_df["C_sr"] = (all_df["Close"] <= all_df["Wall_Sup"] * 1.015).astype(int) - (all_df["Close"] >= all_df["Wall_Res"] * 0.985).astype(int)
all_df["C_rsi"] = (all_df["RSI_14"] <= rsi_buy_floor).astype(int) - (all_df["RSI_14"] >= rsi_sell_ceil).astype(int)
all_df["C_div"] = all_df["DIV_bull"].astype(int) - all_df["DIV_bear"].astype(int)
all_df["CONF"] = (all_df[["C_sr", "C_rsi", "C_div"]].sum(axis=1)).where(_ok)

anchor_end = len(all_df) - replay_bars_back
hist = all_df.iloc[:anchor_end]                       # everything the system is allowed to know
outcome_df = all_df.iloc[anchor_end:] if replay_bars_back > 0 else pd.DataFrame()
df = hist.tail(int(show_bars)).copy()
last = hist.iloc[-1]
current_market_close = float(last["Close"])

structural_resistance, structural_support = float(last["Wall_Res"]), float(last["Wall_Sup"])
current_rsi = float(last["RSI_14"])
atr = float(last["ATR"])
if np.isnan(structural_resistance) or np.isnan(structural_support) or np.isnan(current_rsi):
    st.error("Not enough history for this timeframe to compute structural levels.")
    st.stop()

# ------------------------------------------------------------------ news (live only: historical headlines are not available, so replay would leak the present)
if replay_bars_back > 0:
    news_multiplier, news = 1.0, {"bull": 0, "bear": 0, "n": 0, "ok": False}
    sentiment_label = "🟨 NEWS SENTIMENT DISABLED IN REPLAY"
else:
    news = cached_news(selected_ticker)
    news_multiplier = news["coefficient"]
    sentiment_label = {1.2: "🟩 LIVE NEWS SENTIMENT: BULLISH DEMAND", 0.8: "🟥 LIVE NEWS SENTIMENT: BEARISH SUPPLY"}.get(
        news_multiplier, "🟨 LIVE NEWS SENTIMENT: NEUTRAL FLOW" if news["ok"] else "🟨 NEWS UNAVAILABLE (neutral)")

# ------------------------------------------------------------------ signal (one function, used live, in replay and in the history statistics)
def decide(close, rsi, sup, res, sma, lma, zl, sd, md, news, atr, dcl=np.nan, dcu=np.nan, divb=False, divs=False, H=15, improved=False):
    """Wall reversals + trend filter. BOS / CHoCH marks are NOT used here.
    improved=False: old rules (target = far wall, stop = wall -/+ max(3%, 1.5 ATR)).
    improved=True : a fresh RSI divergence near a wall also triggers the entry; stop = wall -/+ 1.5 ATR (min 0.25% of price);
                    target = nearest of {wall, Donchian edge, ATR reach for H bars}, so it is a level price can plausibly reach in time."""
    tag = None
    stop_pad = lambda w: max(1.5 * atr, 0.0025 * w) if improved else max(0.03 * w, 1.5 * atr)
    near_b = close <= sup * 1.015 or (improved and divb and close <= sup * 1.03)
    near_s = close >= res * 0.985 or (improved and divs and close >= res * 0.97)
    if near_b and (rsi <= rsi_buy_floor or news > 1.0 or (improved and divb)):
        label, side, tp, sl = "BUY REVERSAL ENTRY", 1, res, sup - stop_pad(sup)
    elif near_s and (rsi >= rsi_sell_ceil or news < 1.0 or (improved and divs)):
        label, side, tp, sl = "SELL REVERSAL ENTRY", -1, sup, res + stop_pad(res)
    elif news > 1.0:
        label, side, tp, sl = "CONSOLIDATION - HOLD LONG", 1, res, sup
    elif news < 1.0:
        label, side, tp, sl = "CONSOLIDATION - HOLD SHORT", -1, sup, res
    elif sma >= lma and close >= zl and sd > 0 and md > 0:
        label, side, tp, sl, tag = "UP TREND", 1, res, zl, "UP"
    elif sma < lma and close < zl and sd < 0 and md < 0:
        label, side, tp, sl, tag = "DOWN TREND", -1, sup, zl, "DOWN"
    elif close > zl:
        label, side, tp, sl = "CONSOLIDATION - HOLD LONG", 1, res, sup
    else:
        label, side, tp, sl = "CONSOLIDATION - HOLD SHORT", -1, sup, res
    if improved and np.isfinite(atr) and atr > 0:
        reach = 1.5 * atr * np.sqrt(max(int(H), 1))
        if side > 0:
            cands = [tp, close + reach] + ([dcu] if np.isfinite(dcu) and dcu >= close + 0.5 * atr else [])
            tp = min(c for c in cands if c > close) if any(c > close for c in cands) else tp
        else:
            cands = [tp, close - reach] + ([dcl] if np.isfinite(dcl) and dcl <= close - 0.5 * atr else [])
            tp = max(c for c in cands if c < close) if any(c < close for c in cands) else tp
    adj = False   # target/stop on the wrong side of price (breakouts) -> ATR levels
    if side > 0:
        if tp <= close: tp, adj = close + 2 * atr, True
        if sl >= close: sl, adj = close - 1.5 * atr, True
    else:
        if tp >= close: tp, adj = close - 2 * atr, True
        if sl <= close: sl, adj = close + 1.5 * atr, True
    return label, side, tp, sl, adj, tag


def _flag(v):
    return bool(v) if pd.notna(v) else False


trend_direction, side, predicted_price, stop_loss_price, levels_adjusted, _tag = decide(
    current_market_close, current_rsi, structural_support, structural_resistance,
    float(last["Short_MA"]), float(last["Long_MA"]), float(last["ZLEMA"]),
    float(last["Short_Diff"]), float(last["Medium_Diff"]), news_multiplier, atr,
    dcl=float(last["DC_Lower"]), dcu=float(last["DC_Upper"]), divb=_flag(last["DIV_bull"]), divs=_flag(last["DIV_bear"]),
    H=forecast_lead_units, improved=improved_rules)
if _tag == "UP":
    sentiment_label = "🟩 ALGO FILTER: TECHNICAL BREAKOUT UP"
elif _tag == "DOWN":
    sentiment_label = "🟥 ALGO FILTER: TECHNICAL BREAKDOWN"
take_profit_price = predicted_price
macro_return_pct = (predicted_price - current_market_close) / current_market_close * 100

# ------------------------------------------------------------------ header metrics
with metrics_slot:
    col1, col2, col3 = st.columns(3)
    col1.metric("Target Asset Ticker", selected_ticker)
    col2.metric("Structural Mode Stance", trend_direction)
    col2.caption(sentiment_label)
    col3.metric("Replay Anchor Target" if replay_bars_back else "Structural Target", f"{cs}{predicted_price:.2f}", f"{macro_return_pct:+.2f}% to target")
    if levels_adjusted:
        st.caption("ℹ️ Price is outside its prior structural wall, so target/stop use ATR-based levels.")

# ------------------------------------------------------------------ neural model panel
st.markdown("### 🧠 Neural Model (LSTM) - P(price higher in 4 trading days)")
if replay_bars_back > 0:
    st.info("Neural panel is disabled during replay: the saved model was trained on data that includes the replayed period, so it would not be a fair test. Use `python backtester.py TICKER` for an out-of-sample test.")
else:
    snap = neural_snapshot(selected_ticker)
    if "error" in snap and importlib.util.find_spec("tensorflow") is None:
        st.info("The neural model is switched off on this deployment (TensorFlow is not installed, to fit a small free server). "
                "Everything else - structure marks, Donchian, RSI divergence trades, confluence probability and the hit-rate tables - works normally. "
                "Run the full version on your PC or a 2 GB+ server to use the LSTM.")
    elif "error" in snap:
        st.warning(f"No up-to-date model for {selected_ticker}. Train one: `python train_model.py {selected_ticker}`")
        if st.button(f"Train {selected_ticker} model now (about 1-2 min)"):
            with st.spinner("Training..."):
                try:
                    from generate_forecast import train_specific_ticker
                    train_specific_ticker(selected_ticker)
                    neural_snapshot.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
    else:
        m = snap["meta"]
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("P(UP)", f"{snap['prob_up']:.1%}", snap["direction"])
        n2.metric("Expected price (est.)", f"{cs}{snap['expected_price']:.2f}", f"1σ {cs}{snap['band_low']:.0f}-{cs}{snap['band_high']:.0f}", delta_color="off")
        n3.metric("Validation accuracy", f"{m['val_accuracy']:.1%}", f"baseline {m['val_baseline']:.1%}", delta_color="off")
        n4.metric("Validated edge", "YES" if snap["has_edge"] else "NO", f"n={m['n_val']} samples", delta_color="off")
        if not snap["has_edge"]:
            st.warning("This model did not beat the always-guess-the-majority baseline on unseen data. Treat its signal as noise.")
        if snap["stale"]:
            st.warning(f"Model input data is from {snap['as_of']} - older than 5 days.")
        st.caption(f"Data through {snap['as_of']} · model trained {m['trained_at']} · the expected price is a volatility-scaled estimate, not a direct model output.")

st.markdown("---")

# ------------------------------------------------------------------ confluence + HISTORICAL probability
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


H = int(forecast_lead_units)
_t = pd.DataFrame({"score": hist["CONF"], "ret": hist["Close"].shift(-H) / hist["Close"] - 1}).dropna()   # only outcomes already known at the anchor
cur_score = int(hist["CONF"].iloc[-1]) if pd.notna(hist["CONF"].iloc[-1]) else 0
st.markdown("### 🎯 Confluence & Historical Probability  (Support/Resistance · RSI · RSI Divergence)")
if len(_t) < 40:
    st.info("Not enough history on this timeframe to estimate probabilities.")
else:
    base = float((_t.ret > 0).mean())
    sel = _t[_t.score == cur_score]
    n_s, k_s = len(sel), int((sel.ret > 0).sum())
    p_shr = (k_s + 20 * base) / (n_s + 20)                       # shrunk toward the base rate: small samples cannot claim big edges
    lo, hi = wilson(k_s, n_s)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Confluence score", f"{cur_score:+d}", "bullish" if cur_score > 0 else "bearish" if cur_score < 0 else "neutral", delta_color="off")
    q2.metric(f"P(higher in {H} bars)", f"{p_shr:.0%}", f"base rate {base:.0%} · n={n_s}", delta_color="off")
    q3.metric("Edge vs base rate", f"{(p_shr - base) * 100:+.1f} pts")
    q4.metric("95% range (raw)", f"{lo:.0%} - {hi:.0%}" if n_s else "n/a")
    lc = hist.iloc[-1]
    parts = [("Support / resistance", int(lc["C_sr"]), "near support" if lc["C_sr"] > 0 else "near resistance" if lc["C_sr"] < 0 else "mid-range"),
             ("RSI", int(lc["C_rsi"]), f"RSI {current_rsi:.1f}" + (" (oversold zone)" if lc["C_rsi"] > 0 else " (overbought zone)" if lc["C_rsi"] < 0 else "")),
             ("RSI divergence", int(lc["C_div"]), "bullish divergence" if lc["C_div"] > 0 else "bearish divergence" if lc["C_div"] < 0 else "none active")]
    st.markdown("  \n".join(f"- **{nm}:** {v:+d} - {txt}" for nm, v, txt in parts))
    if n_s < 30:
        st.warning(f"Only {n_s} past bars had this exact score - treat the probability as weak evidence.")
    st.caption("This is the HISTORICAL frequency of the price being higher after the same confluence score on this ticker/timeframe "
               "(only outcomes known before the anchor bar), shrunk toward the base rate. It is not a forecast guarantee; overlapping bars make samples look larger than they are.")
    with st.expander("Probability by confluence score (this ticker & timeframe)"):
        rows_ = []
        for sc, g in _t.groupby("score"):
            rows_.append({"Score": f"{int(sc):+d}", "Bars": len(g), f"P(higher in {H})": f"{(g.ret > 0).mean():.0%}", "Avg return": f"{g.ret.mean() * 100:+.2f}%"})
        for nm, g in (("≥ +2 (strong bullish)", _t[_t.score >= 2]), ("≤ -2 (strong bearish)", _t[_t.score <= -2]), ("ALL BARS (base rate)", _t)):
            if len(g):
                rows_.append({"Score": nm, "Bars": len(g), f"P(higher in {H})": f"{(g.ret > 0).mean():.0%}", "Avg return": f"{g.ret.mean() * 100:+.2f}%"})
        st.table(pd.DataFrame(rows_).set_index("Score"))
        st.caption("If the bullish rows (+2 and above) are not clearly above the base rate - and the bearish rows clearly below it - the combination has no edge here.")

st.markdown("---")

# ------------------------------------------------------------------ RSI divergence @ support / resistance: entries & exits
def div_sr_trades(d, n, tol_pct, window, rr, long_only_):
    """BUY setup  = bullish RSI divergence whose swing low sits AT support ('DIV @ SUPPORT'), or a divergence that formed earlier
    and is still valid when price later touches the support line ('SUPPORT AFTER DIV').
    SELL setup = the mirror image at resistance. A SELL setup EXITS a long (and opens a short unless long-only); a BUY setup
    covers a short. Protective exits: STOP beyond the divergence swing (+0.5 ATR, min 1 ATR) and TARGET at `rr` x risk;
    stop is assumed first if both are touched in one bar. Entries/exits at the signal bar's close (stops/targets at their level)."""
    t = tol_pct / 100.0
    H, L, C = d["High"].values, d["Low"].values, d["Close"].values
    sup, res, atr = d["Wall_Sup"].values, d["Wall_Res"].values, d["ATR"].values
    evb, evs, pxb, pxs = d["DIV_ev_bull"].values, d["DIV_ev_bear"].values, d["DIV_ev_bull_px"].values, d["DIV_ev_bear_px"].values
    pos, cur, armed_b, armed_s, rows = 0, None, None, None, []

    def close_trade(i, px, reason):
        nonlocal pos, cur
        rows.append(dict(side=pos, e_i=cur["i"], x_i=i, entry=cur["entry"], exit=px, ret=pos * (px / cur["entry"] - 1),
                         kind=cur["kind"], reason=reason, open=False))
        pos, cur = 0, None

    for i in range(len(d)):
        exited = False
        if pos != 0:
            if pos > 0 and L[i] <= cur["stop"]:      close_trade(i, cur["stop"], "STOP"); exited = True
            elif pos > 0 and H[i] >= cur["tgt"]:     close_trade(i, cur["tgt"], "TARGET"); exited = True
            elif pos < 0 and H[i] >= cur["stop"]:    close_trade(i, cur["stop"], "STOP"); exited = True
            elif pos < 0 and L[i] <= cur["tgt"]:     close_trade(i, cur["tgt"], "TARGET"); exited = True
        buy = sell = None
        k = i - n
        if evb[i] == 1:
            if k >= 0 and not np.isnan(sup[k]) and L[k] <= sup[k] * (1 + t):
                buy = ("DIV @ SUPPORT", pxb[i])
            else:
                armed_b = (pxb[i], i)
        if evs[i] == 1:
            if k >= 0 and not np.isnan(res[k]) and H[k] >= res[k] * (1 - t):
                sell = ("DIV @ RESISTANCE", pxs[i])
            else:
                armed_s = (pxs[i], i)
        if armed_b:
            if C[i] < armed_b[0] or i - armed_b[1] > window:
                armed_b = None
            elif i > armed_b[1] and not np.isnan(sup[i]) and L[i] <= sup[i] * (1 + t):
                buy, armed_b = ("SUPPORT AFTER DIV", armed_b[0]), None
        if armed_s:
            if C[i] > armed_s[0] or i - armed_s[1] > window:
                armed_s = None
            elif i > armed_s[1] and not np.isnan(res[i]) and H[i] >= res[i] * (1 - t):
                sell, armed_s = ("RESISTANCE AFTER DIV", armed_s[0]), None
        if sell and pos > 0:
            close_trade(i, C[i], "EXIT: " + sell[0]); exited = True
        if buy and pos < 0:
            close_trade(i, C[i], "COVER: " + buy[0]); exited = True
        a = atr[i]
        if pos == 0 and not exited and not np.isnan(a):
            if buy:
                risk = max(C[i] - (buy[1] - 0.5 * a), a)
                if risk <= 4 * a:                                   # skip stale setups where price already ran away
                    pos, cur = 1, dict(i=i, entry=C[i], stop=C[i] - risk, tgt=C[i] + rr * risk, kind="BUY: " + buy[0])
            elif sell and not long_only_:
                risk = max((sell[1] + 0.5 * a) - C[i], a)
                if risk <= 4 * a:
                    pos, cur = -1, dict(i=i, entry=C[i], stop=C[i] + risk, tgt=C[i] - rr * risk, kind="SELL: " + sell[0])
    if pos != 0:
        rows.append(dict(side=pos, e_i=cur["i"], x_i=len(d) - 1, entry=cur["entry"], exit=C[-1], ret=pos * (C[-1] / cur["entry"] - 1),
                         kind=cur["kind"], reason="OPEN", open=True))
    return pd.DataFrame(rows, columns=["side", "e_i", "x_i", "entry", "exit", "ret", "kind", "reason", "open"])


tr_div = div_sr_trades(hist, swing_n, div_tol, div_window, bos_rr, long_only) if show_div_sr else pd.DataFrame()


# ------------------------------------------------------------------ chart helper script (runs inside the chart page)
CHART_JS = r"""
var gd = document.getElementById('{plot_id}');
var BG='__BG__', FG='__FG__', EDGE='__EDGE__';
function fmt(v){var a=Math.abs(v); return a>=10 ? v.toFixed(2) : (a>=1 ? v.toFixed(3) : v.toFixed(5));}
// 1) live price label on the left price axis, exactly where the horizontal cursor line meets it
var tag = document.createElement('div');
tag.style.cssText='position:absolute;pointer-events:none;z-index:50;display:none;font:700 13px sans-serif;padding:2px 7px;border-radius:3px;transform:translate(-100%,-50%);white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.4);background:#FFD600;color:#111;';
gd.style.position='relative'; gd.appendChild(tag);
gd.addEventListener('mousemove', function(e){
  var fl = gd._fullLayout; if(!fl) return;
  var r = gd.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  var xa = fl.xaxis, shown = false;
  if (x >= xa._offset && x <= xa._offset + xa._length) {
    for (var k in fl) {
      if (!/^yaxis\d*$/.test(k)) continue;
      var ya = fl[k];
      if (ya.showticklabels === false && k !== 'yaxis') continue;      // skip the hidden volume strip
      if (y >= ya._offset && y <= ya._offset + ya._length) {
        var v = ya.p2l(y - ya._offset);
        tag.textContent = fmt(v); tag.style.left = (xa._offset - 3) + 'px'; tag.style.top = y + 'px'; tag.style.display='block'; shown = true; break;
      }
    }
  }
  if (!shown) tag.style.display='none';
});
gd.addEventListener('mouseleave', function(){ tag.style.display='none'; });

// 2) price tags on lines / rectangles / circles you draw with the toolbar (updated when you draw, move or resize them)
var busy=false, lastSig='';
function refreshLabels(){
  if (busy) return;
  var L = gd.layout, shapes = L.shapes || [], keep = (L.annotations || []).filter(function(a){return a.name !== 'tp_shape_lbl';});
  var labs = [];
  shapes.forEach(function(s){
    if (!s || s.name === 'tp_user' || (s.type !== 'line' && s.type !== 'rect' && s.type !== 'circle')) return;
    if (!s.yref || s.yref === 'paper') return;
    var pts = (s.type === 'line') ? [[s.x0, s.y0], [s.x1, s.y1]] : [[s.x1, s.y0], [s.x1, s.y1]];
    pts.forEach(function(p){
      labs.push({name:'tp_shape_lbl', xref:s.xref, yref:s.yref, x:p[0], y:p[1], text:'<b>'+fmt(+p[1])+'</b>', showarrow:false,
                 xanchor:'left', xshift:6, bgcolor:'#FFD600', font:{color:'#111', size:12}, borderpad:2});
    });
  });
  var sig = JSON.stringify(labs.map(function(a){return [a.x,a.y,a.yref];}));
  if (sig === lastSig) return;
  lastSig = sig; busy = true;
  Plotly.relayout(gd, {annotations: keep.concat(labs)}).then(function(){busy=false;}, function(){busy=false;});
}
gd.on('plotly_relayout', function(){ setTimeout(refreshLabels, 0); });
"""

# ------------------------------------------------------------------ chart
fmt = '%Y-%m-%d %H:%M' if active_cfg["is_intraday"] else '%Y-%m-%d'
timeline_x = list(df.index.strftime(fmt))
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.66, 0.04, 0.16, 0.14])
fig.add_trace(go.Candlestick(x=timeline_x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Market History"), row=1, col=1)

if replay_bars_back > 0 and not outcome_df.empty:
    outcome_df = outcome_df.head(int(forecast_lead_units))   # only the forecast horizon is ever used, so history keeps its width
if replay_bars_back > 0 and not outcome_df.empty and reveal_future:
    outcome_x = list(outcome_df.index.strftime(fmt))
    total_axis_x = timeline_x + outcome_x
    fig.add_trace(go.Candlestick(x=outcome_x, open=outcome_df["Open"], high=outcome_df["High"], low=outcome_df["Low"],
                                 close=outcome_df["Close"], name="Replay Outcome", opacity=0.35), row=1, col=1)
    proj_target_x = outcome_x[min(forecast_lead_units, len(outcome_x)) - 1]
else:
    future_labels = [f"Forecast Step +{i}" for i in range(1, forecast_lead_units + 1)]
    total_axis_x = timeline_x + future_labels
    proj_target_x = future_labels[-1]

def hline(y, name, color, width=2, dash="dash", tag=None):
    fig.add_trace(go.Scatter(x=timeline_x, y=[y] * len(df), mode="lines", name=name, hoverinfo="skip", line=dict(color=color, width=width, dash=dash)), row=1, col=1)
    if tag:   # exact price written at the right edge of the line
        fig.add_annotation(xref="paper", yref="y", x=1, y=y, text=f"<b>{tag}</b> {y:,.2f}", showarrow=False, xanchor="left", font=dict(color=color, size=13))

hline(structural_resistance, "Structural Resistance", "#FF007F", tag="RES")
hline(structural_support, "Structural Support", "#00FFCC", tag="SUP")
hline(stop_loss_price, "Risk Stop", "#FF9100", 2.5, "dot", tag="STOP")
hline(take_profit_price, "Target Profit", "#00B0FF", 2.5, "dot", tag="TGT")
if toggle_bb:
    for col in ("BB_Upper", "BB_Lower"):
        fig.add_trace(go.Scatter(x=timeline_x, y=df[col], mode="lines", line=dict(color=bb_color, width=bb_width), name=col.replace("_", " ")), row=1, col=1)
if toggle_zlema:
    fig.add_trace(go.Scatter(x=timeline_x, y=df["ZLEMA"], mode="lines", line=dict(color="#00ffbb", width=2, dash="longdash"), name="Zero Lag Baseline"), row=1, col=1)
if toggle_dc:
    _edge = dc_style.startswith("Edge")
    for col, nm in (("DC_Upper", "Donchian Upper"), ("DC_Lower", "Donchian Lower")):
        fig.add_trace(go.Scatter(x=timeline_x, y=df[col], mode="lines", line=dict(color="#FFD600", width=1.5, shape="hv" if _edge else "linear"), name=nm), row=1, col=1)
    if _edge:
        _pos = {ts: p for p, ts in enumerate(all_df.index)}
        for _bc, _px, _nm in (("DC_up_bar", "High", "Donchian edge (high)"), ("DC_lo_bar", "Low", "Donchian edge (low)")):
            _bars = sorted({int(x) for x in df[_bc].dropna().unique()})
            _bars = [b_ for b_ in _bars if all_df.index[b_] in _pos and all_df.index[b_] >= df.index[0]]
            if _bars:
                fig.add_trace(go.Scatter(x=[all_df.index[b_].strftime(fmt) for b_ in _bars], y=[float(all_df[_px].iloc[b_]) for b_ in _bars], mode="markers",
                                         marker=dict(symbol="diamond-open", size=10, color="#FFD600", line=dict(width=2)), name=_nm), row=1, col=1)
_pivot_levels = []
if toggle_pivots:
    # P = gold solid; resistances R1-R3 = warm reds (dash -> dashdot -> dot); supports S1-S3 = greens (same dashes). Price tag at the right edge.
    _pv_style = {"P": ("#FFC400", 2.2, "solid"),
                 "R1": ("#FF8A65", 1.8, "dash"), "R2": ("#F44336", 1.8, "dashdot"), "R3": ("#C62828", 1.8, "dot"),
                 "S1": ("#4DB6AC", 1.8, "dash"), "S2": ("#00C853", 1.8, "dashdot"), "S3": ("#2E7D32", 1.8, "dot")}
    for name, val in compute_institutional_pivots(hist.tail(20), pivot_mode).items():
        _pc, _pw, _pd = _pv_style.get(name, ("#B0BEC5", 1.5, "dot"))
        hline(val, f"Pivot {name}", _pc, _pw, _pd)
        fig.add_annotation(xref="paper", yref="y", x=1, y=val, text=f"<b>{name}</b> {val:,.2f}", showarrow=False, xanchor="left",
                           font=dict(color=_pc, size=13))
        _pivot_levels.append(val)
if show_bos:
    _ev = df[df["BOS_event"] != 0]
    for _dir, _sym, _clr, _nm, _ypos in ((1, "triangle-up", "#00E676", "Bullish BOS/CHoCH", "Low"), (-1, "triangle-down", "#FF5252", "Bearish BOS/CHoCH", "High")):
        _e = _ev[_ev["BOS_event"] == _dir]
        if len(_e):
            fig.add_trace(go.Scatter(x=list(_e.index.strftime(fmt)), y=_e[_ypos] + (-0.6 if _dir > 0 else 0.6) * _e["ATR"].fillna(0), mode="markers+text",
                                     text=["CHoCH" if c_ else "BOS" for c_ in _e["BOS_event_choch"]], textposition="bottom center" if _dir > 0 else "top center",
                                     marker=dict(symbol=_sym, size=13, color=_clr), name=_nm), row=1, col=1)
    _segx, _segy = [], []                       # short line from the swing that formed the broken level to the break bar
    _vis_set = set(df.index)
    for ts_, r_ in _ev.iterrows():
        lb_ = int(r_["BOS_level_bar"])
        start_ = all_df.index[lb_] if all_df.index[lb_] >= df.index[0] else df.index[0]
        _segx += [start_.strftime(fmt), ts_.strftime(fmt), None]
        _segy += [float(r_["BOS_level"]), float(r_["BOS_level"]), None]
    if _segx:
        fig.add_trace(go.Scatter(x=_segx, y=_segy, mode="lines", line=dict(color="rgba(255,255,255,0.55)" if template == "plotly_dark" else "rgba(30,41,59,0.6)", width=1, dash="dot"),
                                 name="Broken level", connectgaps=False), row=1, col=1)
if show_div_sr and len(tr_div):
    _idx, _vis = hist.index, set(df.index)
    _ent = {1: ([], [], []), -1: ([], [], [])}
    _ext = {"STOP": ([], [], []), "TARGET": ([], [], []), "SETUP": ([], [], [])}
    _seg = {True: ([], []), False: ([], [])}
    for _, r_ in tr_div.iterrows():
        ei, xi = int(r_.e_i), int(r_.x_i)
        ex_lbl, xx_lbl = _idx[ei].strftime(fmt), _idx[xi].strftime(fmt)
        if _idx[ei] in _vis:
            _ent[int(r_.side)][0].append(ex_lbl)
            _ent[int(r_.side)][1].append(float(hist["Low"].iloc[ei] - 1.8 * np.nan_to_num(hist["ATR"].iloc[ei])) if r_.side > 0 else float(hist["High"].iloc[ei] + 1.8 * np.nan_to_num(hist["ATR"].iloc[ei])))
            _ent[int(r_.side)][2].append(r_.kind.replace(": ", "<br>"))
        if not r_.open and _idx[xi] in _vis:
            key = "STOP" if r_.reason == "STOP" else "TARGET" if r_.reason == "TARGET" else "SETUP"
            _ext[key][0].append(xx_lbl); _ext[key][1].append(float(r_.exit)); _ext[key][2].append(r_.reason.replace(": ", "<br>"))
        if (_idx[ei] in _vis) and (_idx[xi] in _vis):
            _seg[bool(r_.ret > 0)][0].extend([ex_lbl, xx_lbl, None]); _seg[bool(r_.ret > 0)][1].extend([float(r_.entry), float(r_.exit), None])
    for win_, clr_, nm_ in ((True, "rgba(0,230,118,0.8)", "Div trade (win)"), (False, "rgba(255,82,82,0.8)", "Div trade (loss)")):
        if _seg[win_][0]:
            fig.add_trace(go.Scatter(x=_seg[win_][0], y=_seg[win_][1], mode="lines", line=dict(color=clr_, width=2, dash="dot"), name=nm_, connectgaps=False), row=1, col=1)
    for side_, sym_, clr_, nm_, pos_ in ((1, "triangle-up", "#00E676", "BUY (RSI div @ support)", "bottom center"), (-1, "triangle-down", "#FF1744", "SELL (RSI div @ resistance)", "top center")):
        if _ent[side_][0]:
            fig.add_trace(go.Scatter(x=_ent[side_][0], y=_ent[side_][1], mode="markers+text", text=_ent[side_][2], textposition=pos_, textfont=dict(size=11, color=clr_),
                                     marker=dict(symbol=sym_, size=18, color=clr_, line=dict(color="#FFFFFF", width=1.5)), name=nm_), row=1, col=1)
    for key_, clr_, nm_ in (("STOP", "#FF5252", "Exit: stop"), ("TARGET", "#00E676", "Exit: target"), ("SETUP", "#FFB300", "Exit: opposite S/R setup")):
        if _ext[key_][0]:
            fig.add_trace(go.Scatter(x=_ext[key_][0], y=_ext[key_][1], mode="markers+text", text=_ext[key_][2], textposition="middle right", textfont=dict(size=11, color=clr_),
                                     marker=dict(symbol="x", size=15, color=clr_, line=dict(width=3, color=clr_)), name=nm_), row=1, col=1)
fig.add_trace(go.Scatter(x=[timeline_x[-1], proj_target_x], y=[current_market_close, predicted_price], mode="lines+markers", name="Structural Target",
                         line=dict(color="#00B0FF", width=4, dash="dash"), marker=dict(size=14, symbol="diamond", color="#FFFFFF")), row=1, col=1)

bar_colors = ["#00E676" if s >= l else "#FF5252" for s, l in zip(df["Short_MA"], df["Long_MA"])]
fig.add_trace(go.Bar(x=timeline_x, y=[1.0] * len(df), marker=dict(color=bar_colors, line=dict(width=0)), showlegend=False, hoverinfo="skip", width=active_cfg["bar_width"]), row=2, col=1)
for col, nm, colr in (("Long_Diff", "Long Diff", "#E040FB"), ("Medium_Diff", "Medium Diff", "#00B0FF"), ("Short_Diff", "Short Diff", "#00E676")):
    fig.add_trace(go.Scatter(x=timeline_x, y=df[col], mode="lines", name=nm, line=dict(color=colr, width=3)), row=3, col=1)
fig.add_trace(go.Scatter(x=timeline_x, y=df["RSI_14"], mode="lines", name="RSI (14)", line=dict(color="#B388FF", width=2.5)), row=4, col=1)
for _lvl in (rsi_buy_floor, rsi_sell_ceil):
    fig.add_trace(go.Scatter(x=timeline_x, y=[_lvl] * len(df), mode="lines", showlegend=False, hoverinfo="skip", line=dict(color="#FF9100", width=1, dash="dot")), row=4, col=1)
for _dir, _sym, _clr, _col, _ypos, _mult, _nm in ((1, "diamond", "#00E676", "DIV_ev_bull", "Low", -1.1, "Bullish RSI divergence"),
                                                   (-1, "diamond", "#FF5252", "DIV_ev_bear", "High", 1.1, "Bearish RSI divergence")):
    _d = df[df[_col] != 0]
    if len(_d):
        fig.add_trace(go.Scatter(x=list(_d.index.strftime(fmt)), y=_d[_ypos] + _mult * _d["ATR"].fillna(0), mode="markers+text", text=["RSI Div"] * len(_d), textfont=dict(size=11),
                                 textposition="bottom center" if _dir > 0 else "top center", marker=dict(symbol=_sym, size=11, color=_clr), name=_nm), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(_d.index.strftime(fmt)), y=_d["RSI_14"], mode="markers", showlegend=False, marker=dict(symbol=_sym, size=10, color=_clr)), row=4, col=1)

# ---- user drawing tools (sidebar): horizontal / vertical / cross lines, kept per ticker + timeframe
st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 Chart Tools")
st.session_state.setdefault("tp_lines", [])
st.session_state.setdefault("tp_gid", 0)
tool_kind = st.sidebar.selectbox("Line to add", ["Horizontal line", "Vertical line", "Cross lines (H + V)"])
tool_price = st.sidebar.number_input("Price level (blank = last close)", value=None, placeholder=f"{current_market_close:.2f}", format="%.4f")
tool_back = st.sidebar.number_input("Vertical line: bars back from latest candle", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)
_c1, _c2 = st.sidebar.columns(2)
tool_color = _c1.color_picker("Colour", "#FFD600")
tool_width = _c2.slider("Width", 1, 5, 2)
tool_dash = st.sidebar.selectbox("Style", ["dash", "solid", "dot", "dashdot"])
_b1, _b2, _b3 = st.sidebar.columns(3)
if _b1.button("Add"):
    st.session_state["tp_gid"] += 1
    _y = float(tool_price) if tool_price is not None else float(current_market_close)
    _x = df.index[-1 - int(tool_back)].strftime(fmt)
    for _k in (["H"] if tool_kind.startswith("Hor") else ["V"] if tool_kind.startswith("Ver") else ["H", "V"]):
        st.session_state["tp_lines"].append(dict(gid=st.session_state["tp_gid"], kind=_k, y=_y, x=_x, tk=selected_ticker, tf=selected_tf,
                                                 color=tool_color, width=tool_width, dash=tool_dash))
_mine = [l for l in st.session_state["tp_lines"] if l["tk"] == selected_ticker and l["tf"] == selected_tf]
if _b2.button("Undo") and _mine:
    _g = max(l["gid"] for l in _mine)
    st.session_state["tp_lines"] = [l for l in st.session_state["tp_lines"] if not (l["tk"] == selected_ticker and l["tf"] == selected_tf and l["gid"] == _g)]
if _b3.button("Clear"):
    st.session_state["tp_lines"] = [l for l in st.session_state["tp_lines"] if not (l["tk"] == selected_ticker and l["tf"] == selected_tf)]
_mine = [l for l in st.session_state["tp_lines"] if l["tk"] == selected_ticker and l["tf"] == selected_tf]
st.sidebar.caption(f"{len(_mine)} line(s) on this chart. Use the toolbar above the chart for free-hand lines, rectangles and circles (drag them to move, eraser to delete).")

_user_levels = []
for ln in _mine:
    _dsh = dict(color=ln["color"], width=ln["width"], dash=ln["dash"])
    if ln["kind"] == "H":
        fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1, y0=ln["y"], y1=ln["y"], line=_dsh, layer="above", name="tp_user")
        fig.add_annotation(xref="paper", yref="y", x=1, y=ln["y"], text=f"{ln['y']:,.2f}", showarrow=False, xanchor="left", font=dict(color=ln["color"], size=13))
        _user_levels.append(ln["y"])
    elif ln["x"] in timeline_x:
        fig.add_shape(type="line", xref="x", yref="paper", x0=ln["x"], x1=ln["x"], y0=0, y1=1, line=_dsh, layer="above", name="tp_user")

# ---- price axis fitted to the candles (+ nearby key levels), so nothing far away squashes the chart
if fit_y:
    _lo, _hi = float(df["Low"].min()), float(df["High"].max())
    if replay_bars_back > 0 and not outcome_df.empty and reveal_future:
        _o = outcome_df
        _lo, _hi = min(_lo, float(_o["Low"].min())), max(_hi, float(_o["High"].max()))
    _rng = max(_hi - _lo, 1e-9)
    _c_lo, _c_hi = _lo, _hi
    for _v in [structural_resistance, structural_support, stop_loss_price, take_profit_price, predicted_price] + _user_levels:
        if np.isfinite(_v) and (_c_lo - 0.6 * _rng) <= _v <= (_c_hi + 0.6 * _rng):
            _lo, _hi = min(_lo, _v), max(_hi, _v)
    _c_lo2, _c_hi2 = _lo, _hi
    for _v in _pivot_levels:      # pivots only stretch the axis when they sit close to the candles
        if (_c_lo2 - 0.25 * _rng) <= _v <= (_c_hi2 + 0.25 * _rng):
            _lo, _hi = min(_lo, _v), max(_hi, _v)
    _pad = max(0.05 * (_hi - _lo), 2.6 * np.nan_to_num(atr))
    fig.update_yaxes(range=[_lo - _pad, _hi + _pad], row=1, col=1)

fig.update_layout(template=template, height=1400, xaxis_rangeslider_visible=False, paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                  margin=dict(l=50, r=130, t=150, b=50), font=dict(size=18, color=text_color),
                  legend=dict(font=dict(size=14, color=text_color), orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
                  hovermode="x" if crosshair else "closest",
                  hoverlabel=dict(bgcolor=card_bg, bordercolor=border_color, font=dict(color=text_color, size=13)),
                  dragmode="pan" if chart_drag == "Pan" else "zoom",
                  modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)", color=muted_color, activecolor="#00B0FF"),
                  newshape=dict(line=dict(color="#FFD600", width=2), opacity=1),
                  uirevision=f"{selected_ticker}|{selected_tf}|{show_bars}|{replay_bars_back}")   # keeps zoom/pan/drawings across reruns
fig.update_xaxes(type="category", categoryorder="array", categoryarray=total_axis_x, showgrid=True, gridcolor=grid_color,
                 tickfont=dict(size=13, color=text_color), nticks=18, tickangle=-35)
fig.update_yaxes(showgrid=True, gridcolor=grid_color, tickfont=dict(size=15, color=text_color))
fig.update_yaxes(row=2, col=1, showgrid=False, showticklabels=False)
fig.update_yaxes(row=4, col=1, range=[0, 100], title_text="RSI")
# only candles (and the lower panels) show hover read-outs; overlays/markers stay quiet so tooltips don't pile up
for _tr in fig.data:
    if _tr.type == "scatter" and (_tr.yaxis in (None, "y")):
        if "markers" in str(_tr.mode) or _tr.hoverinfo == "skip":
            _tr.hoverinfo = "skip"
        else:
            _tr.hovertemplate = "%{y:,.2f}  <i>%{fullData.name}</i><extra></extra>"
if crosshair:
    _spike = dict(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="dot", spikecolor=muted_color)
    fig.update_xaxes(**_spike)
    fig.update_yaxes(**_spike)
_cfg = {"scrollZoom": True, "displaylogo": False, "doubleClick": "reset",
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"],
        "edits": {"shapePosition": True},
        "toImageButtonOptions": {"format": "png", "filename": "tradepoint_chart", "scale": 2}}
_js = CHART_JS.replace("__BG__", card_bg).replace("__FG__", text_color).replace("__EDGE__", border_color)
_html = fig.to_html(full_html=True, include_plotlyjs="cdn", config=_cfg, post_script=_js, default_height="1400px", default_width="100%")
_html = _html.replace("</head>", f"<style>html,body{{margin:0;padding:0;background:{bg_color};overflow:hidden}}</style></head>", 1)
with chart_slot:
    if hasattr(st, "iframe"):          # newer Streamlit
        st.iframe(_html, height=1410)
    else:                              # older Streamlit (requirements allow >=1.40)
        st.components.v1.html(_html, height=1410, scrolling=False)

if show_div_sr:
    with st.expander("📋 RSI-divergence @ support/resistance - trade log & statistics", expanded=False):
        if tr_div.empty:
            st.info("No setups found on this timeframe with the current settings.")
        else:
            done = tr_div[~tr_div.open]
            cut = len(hist) * 0.6

            def _sm(g):
                w, lo_ = g[g.ret > 0].ret.sum(), -g[g.ret < 0].ret.sum()
                return pd.Series({"Trades": len(g), "Win rate": f"{(g.ret > 0).mean():.0%}", "Avg return": f"{g.ret.mean() * 100:+.2f}%",
                                  "Profit factor": f"{w / lo_:.2f}" if lo_ > 0 else "-", "Stopped": f"{(g.reason == 'STOP').mean():.0%}",
                                  "Target": f"{(g.reason == 'TARGET').mean():.0%}"})
            summ = {"All closed trades": _sm(done)} if len(done) else {}
            for nm_, g in (("First 60% (in-sample)", done[done.e_i <= cut]), ("Last 40% (OUT-OF-SAMPLE)", done[done.e_i > cut])):
                if len(g):
                    summ[nm_] = _sm(g)
            if summ:
                st.table(pd.DataFrame(summ).T)
            log_ = tr_div.tail(15).copy()
            log_["Entry date"] = [hist.index[int(i)].strftime(fmt) for i in log_.e_i]
            log_["Exit date"] = [hist.index[int(i)].strftime(fmt) if not o else "open" for i, o in zip(log_.x_i, log_.open)]
            log_["Side"] = np.where(log_.side > 0, "LONG", "SHORT")
            log_["Return"] = (log_.ret * 100).map("{:+.2f}%".format)
            log_["Entry"] = log_.entry.map("{:,.2f}".format)
            log_["Exit"] = log_.exit.map("{:,.2f}".format)
            st.table(log_[["Entry date", "Side", "kind", "Entry", "Exit date", "Exit", "reason", "Return"]].rename(columns={"kind": "Setup", "reason": "Exit reason"}).set_index("Entry date"))
            st.caption("Entries/exits at the signal bar's close, stops/targets at their price level (stop assumed first), no fees. "
                       "With few trades these statistics are anecdotes - judge by the out-of-sample row and the trade count.")

# ------------------------------------------------------------------ confluence + replay scorecard
st.markdown("### 🗺️ Institutional Confluence Scoring Matrix")
c1, c2, c3 = st.columns(3)
c1.info(f"**1. Core Structural Bounds ({sr_eff}-bar):**\n\nCeiling Wall: {csm}{structural_resistance:.2f}\n\nFloor Base: {csm}{structural_support:.2f}")
c2.info(f"**2. Trend Strength:**\n\nWilder ADX: {float(last['ADX']):.2f}  ·  RSI(14): {current_rsi:.1f}")
if "ENTRY" in trend_direction:
    c3.success(f"**3. Reversal Signal Status:**\n\n🟢 REVERSAL ZONE DETECTED: {trend_direction}")
else:
    c3.warning(f"**3. Reversal Signal Status:**\n\n⚠️ PRICE MID-CHANNEL: {trend_direction}")

if replay_bars_back > 0 and not outcome_df.empty and reveal_future:
    st.markdown("### 🎯 Automated Replay Performance Scorecard")
    window = outcome_df.head(forecast_lead_units)
    tp_hit = sl_hit = False
    hit_bar = -1
    for idx, (_, r) in enumerate(window.iterrows(), start=1):
        hit_sl = r["Low"] <= stop_loss_price if side > 0 else r["High"] >= stop_loss_price
        hit_tp = r["High"] >= take_profit_price if side > 0 else r["Low"] <= take_profit_price
        if hit_sl:                       # conservative: if both are touched in one bar, assume the stop hit first
            sl_hit, hit_bar = True, idx
            break
        if hit_tp:
            tp_hit, hit_bar = True, idx
            break
    s1, s2 = st.columns(2)
    with s1:
        final_close = float(window["Close"].iloc[-1])
        if tp_hit:
            st.success(f"### 🟩 PREDICTION WIN\nTarget hit at **bar {hit_bar}** of the horizon.")
        elif sl_hit:
            st.error(f"### 🟥 PREDICTION LOSS\nStop breached at **bar {hit_bar}** (stop assumed first when both touched in one bar).")
        elif (final_close - current_market_close) * side > 0:
            st.success(f"### 🟩 TREND WIN (TIMEOUT)\nHorizon closed at **{csm}{final_close:.2f}**, in the signal's favour, without hitting stop or target.")
        else:
            st.info(f"### 🟨 NO RESOLUTION\nHorizon closed at **{csm}{final_close:.2f}** against/flat to the signal, no stop or target hit.")
    with s2:
        fav = float(window["High"].max()) if side > 0 else float(window["Low"].min())
        adv = float(window["Low"].min()) if side > 0 else float(window["High"].max())
        st.markdown(f"**Path Efficiency Metrics:**\n* Max Favorable Excursion: `{(fav / current_market_close - 1) * 100:+.2f}%`\n"
                    f"* Max Adverse Excursion: `{(adv / current_market_close - 1) * 100:+.2f}%`")
    if len(window) < forecast_lead_units:
        st.caption(f"Only {len(window)} of {forecast_lead_units} horizon bars exist after this anchor.")

# ------------------------------------------------------------------ hit-rate over ALL history (one replay is an anecdote; this is the evidence)
def batch_signal_stats(d, horizon, improved=False):
    cols = ["High", "Low", "Close", "Short_MA", "Long_MA", "ZLEMA", "Short_Diff", "Medium_Diff", "ATR", "RSI_14", "Wall_Res", "Wall_Sup"]
    a = {k: d[k].values for k in cols}
    dcl_, dcu_ = d["DC_Lower"].values, d["DC_Upper"].values
    dvb_, dvs_ = d["DIV_bull"].fillna(False).astype(bool).values, d["DIV_bear"].fillna(False).astype(bool).values
    recs = []
    for i in range(len(d) - horizon - 1):
        if any(np.isnan(a[k][i]) for k in cols):
            continue
        label, sd_, tp, sl, _, _ = decide(a["Close"][i], a["RSI_14"][i], a["Wall_Sup"][i], a["Wall_Res"][i], a["Short_MA"][i],
                                          a["Long_MA"][i], a["ZLEMA"][i], a["Short_Diff"][i], a["Medium_Diff"][i], 1.0, a["ATR"][i],
                                          dcl=dcl_[i], dcu=dcu_[i], divb=dvb_[i], divs=dvs_[i], H=horizon, improved=improved)
        entry, exit_px, outcome = a["Close"][i], a["Close"][i + horizon], "timeout"
        for j in range(i + 1, i + horizon + 1):
            hit_sl = a["Low"][j] <= sl if sd_ > 0 else a["High"][j] >= sl
            hit_tp = a["High"][j] >= tp if sd_ > 0 else a["Low"][j] <= tp
            if hit_sl:
                exit_px, outcome = sl, "stop"; break
            if hit_tp:
                exit_px, outcome = tp, "target"; break
        recs.append((i, label, outcome, sd_ * (exit_px / entry - 1), a["Close"][i + horizon] / entry - 1))
    return pd.DataFrame(recs, columns=["i", "label", "outcome", "ret", "raw"])

with st.expander(f"📊 Signal hit-rate over ALL history ({selected_tf}, {forecast_lead_units}-bar horizon, no news)", expanded=True):
    bs = batch_signal_stats(all_df, forecast_lead_units, improved_rules)
    if bs.empty:
        st.info("Not enough history.")
    else:
        def summarize(g):
            return pd.Series({"Signals": len(g), "Target hit": f"{(g.outcome == 'target').mean():.0%}",
                              "Stop hit": f"{(g.outcome == 'stop').mean():.0%}", "Timeout": f"{(g.outcome == 'timeout').mean():.0%}",
                              "Win rate (P&L>0)": f"{(g.ret > 0).mean():.0%}", "Avg return": f"{g.ret.mean() * 100:+.2f}%"})
        tbl = bs.groupby("label").apply(summarize, include_groups=False)
        tbl.loc["ALL STATES"] = summarize(bs)
        ent = bs[bs.label.str.contains("ENTRY")]
        if len(ent):
            cut = bs.i.min() + int((bs.i.max() - bs.i.min()) * 0.6)
            tbl.loc["► REVERSAL ENTRIES ONLY"] = summarize(ent)
            for nm, part in (("   entries: first 60% (in-sample)", ent[ent.i <= cut]), ("   entries: last 40% (OUT-OF-SAMPLE)", ent[ent.i > cut])):
                if len(part):
                    tbl.loc[nm] = summarize(part)
        tbl.loc["Baseline: always long"] = pd.Series({"Signals": len(bs), "Target hit": "-", "Stop hit": "-", "Timeout": "-",
                                                      "Win rate (P&L>0)": f"{(bs.raw > 0).mean():.0%}", "Avg return": f"{bs.raw.mean() * 100:+.2f}%"})
        st.table(tbl)
        st.caption("Entry at the signal bar's close; stop assumed first if both touched in one bar; no news (history unavailable). "
                   "Consecutive signals overlap, so treat counts as indicative. If 'Win rate' is not clearly above 50% and 'Avg return' "
                   "is not clearly positive and above the always-long baseline, this rule has no edge on this ticker/timeframe.")
        st.markdown("**Old rules vs improved rules - reversal entries only (same data, same horizon)**")
        _rows = {}
        for _nm, _flagv in (("Old rules", False), ("Improved rules", True)):
            _b = batch_signal_stats(all_df, forecast_lead_units, _flagv)
            _e = _b[_b.label.str.contains("ENTRY")]
            if _e.empty:
                continue
            _cut = _b.i.min() + int((_b.i.max() - _b.i.min()) * 0.6)
            for _part, _g in (("all", _e), ("OUT-OF-SAMPLE (last 40%)", _e[_e.i > _cut])):
                if len(_g):
                    _rows[f"{_nm} - {_part}"] = summarize(_g)
        if _rows:
            st.table(pd.DataFrame(_rows).T)
            st.caption("Pick the rule set whose OUT-OF-SAMPLE row has the higher win rate AND higher average return with a reasonable number of signals (30+). "
                       "Fewer than ~30 signals means the difference is noise.")

# ------------------------------------------------------------------ real paper portfolio + real rule backtest
st.markdown("### 💼 Paper Portfolio (from local_sandbox_portfolio.json)")
pf = load_portfolio()
shares = pf["holdings"].get(selected_ticker, 0)
avg_cost = pf["avg_cost"].get(selected_ticker)
unreal = (current_market_close - avg_cost) * shares if (shares and avg_cost) else 0.0
p1, p2, p3, p4 = st.columns(4)
p1.metric("Cash", f"{cs}{pf['cash']:,.2f}")
p2.metric(f"{selected_ticker} Shares", f"{shares}")
p3.metric("Avg Entry Cost", f"{cs}{avg_cost:.2f}" if avg_cost else "n/a")
p4.metric("Unrealised P&L", f"{cs}{unreal:+,.2f}" if avg_cost else "n/a")
others = {k: v for k, v in pf["holdings"].items() if v}
if others:
    st.caption("All open positions: " + ", ".join(f"{k}: {v}" for k, v in others.items()))

st.markdown("### 📈 Rule-Based Trend Filter vs Buy & Hold (real prices, signal at close → fill at next open, 5 bps/side)")
st.caption("This tests the simple MA-cross + Zero-Lag rule on the loaded history. It is NOT the neural model - use backtester.py for that.")
bt = hist[["Open", "Close", "Short_MA", "Long_MA", "ZLEMA"]].dropna()
if len(bt) > 60:
    long_sig = ((bt["Short_MA"] >= bt["Long_MA"]) & (bt["Close"] >= bt["ZLEMA"])).astype(float)
    pos = long_sig.shift(1).fillna(0.0)                                   # decided at yesterday's close
    day_ret = np.where(pos > pos.shift(1).fillna(0), bt["Close"] / bt["Open"] - 1,   # entry day: open -> close
                       np.where(pos < pos.shift(1).fillna(0), bt["Open"] / bt["Close"].shift(1) - 1,  # exit day: prev close -> open
                                bt["Close"].pct_change()))
    day_ret = pd.Series(day_ret, index=bt.index) * np.maximum(pos, pos.shift(1).fillna(0))
    day_ret = day_ret - 0.0005 * (pos.diff().abs().fillna(0))
    strat = 10000 * (1 + day_ret.fillna(0)).cumprod()
    hold = 10000 * bt["Close"] / bt["Close"].iloc[0]
    fe = go.Figure()
    fe.add_trace(go.Scatter(x=strat.index, y=strat, name="Trend-filter strategy", line=dict(color="#00E676", width=3)))
    fe.add_trace(go.Scatter(x=hold.index, y=hold, name="Buy & hold", line=dict(color="#00B0FF", width=2, dash="dot")))
    fe.update_layout(template=template, font=dict(color=text_color), height=350, paper_bgcolor=bg_color, plot_bgcolor=bg_color, margin=dict(l=40, r=40, t=10, b=10))
    st.plotly_chart(fe, width="stretch")
    st.caption(f"Strategy {csm}{strat.iloc[-1]:,.0f} vs buy & hold {csm}{hold.iloc[-1]:,.0f} (start {csm}10,000).")

# ------------------------------------------------------------------ scanner
st.markdown("<h3 style='font-size: 32px; margin-top: 40px;'>🔍 Multi-Ticker Neural Scanner</h3>", unsafe_allow_html=True)
run_scan = st.button("Run neural scan on watchlist (uses saved models)")
rows = []
for sym in watchlist:
    w = high_speed_market_download(sym, "10d", "1d")
    if w.empty:
        continue
    price = float(w["Close"].iloc[-1])
    chg = float(w["Close"].pct_change().iloc[-1] * 100) if len(w) > 1 else 0.0
    row = {"Asset": sym, "Last Price": f"{get_currency_symbol(sym)}{price:,.2f}", "Daily %": f"{chg:+.2f}%", "P(UP, 4d)": "-", "Signal": "-", "Validated Edge": "-"}
    if run_scan:
        s = neural_snapshot(sym)
        if "error" in s:
            row["Signal"] = "no model"
        else:
            row["P(UP, 4d)"] = f"{s['prob_up']:.1%}"
            row["Signal"] = ("🟢 UP" if s["direction"] == "UP TREND" else "🔴 DOWN") + ("" if s["has_edge"] else " (no edge)")
            row["Validated Edge"] = "yes" if s["has_edge"] else "no"
    rows.append(row)
if rows:
    st.table(pd.DataFrame(rows))

# ------------------------------------------------------------------ alerts
st.sidebar.markdown("---")
st.sidebar.markdown("### 📧 Alert Email")
user_email = st.sidebar.text_input("Enter Alert Email Address", value="")
if st.sidebar.button("Send stance alert"):
    body = (f"Tradepoint alert for {selected_ticker}: {trend_direction}. Target {cs}{predicted_price:.2f} "
            f"({macro_return_pct:+.2f}%), stop {cs}{stop_loss_price:.2f}.")
    host, user, pw = os.environ.get("VP_SMTP_HOST"), os.environ.get("VP_SMTP_USER"), os.environ.get("VP_SMTP_PASS")
    if not user_email:
        st.sidebar.warning("Enter a destination email.")
    elif not (host and user and pw):
        st.sidebar.warning("Email NOT sent: set VP_SMTP_HOST, VP_SMTP_USER and VP_SMTP_PASS environment variables (optional VP_SMTP_PORT, default 587).")
        st.sidebar.code(body)
    else:
        try:
            msg = MIMEText(body)
            msg["Subject"], msg["From"], msg["To"] = f"Tradepoint signal: {selected_ticker}", user, user_email
            with smtplib.SMTP(host, int(os.environ.get("VP_SMTP_PORT", 587)), timeout=15) as s:
                s.starttls(); s.login(user, pw); s.send_message(msg)
            st.sidebar.success(f"Sent to {user_email}")
        except Exception as e:
            st.sidebar.error(f"Email failed: {e}")

# ------------------------------------------------------------------ replay autoplay: after the page is drawn, wait, advance one bar, redraw
if replay_mode and st.session_state.get("tp_playing_now") and replay_bars_back > 0:
    import time as _time
    _time.sleep(float(st.session_state.get("tp_delay", 1.0)))
    _nxt = replay_bars_back - 1
    st.session_state["tp_pending"] = _nxt
    if _nxt <= 0:
        st.session_state["tp_play"] = False       # reached real-time: stop, like TradingView
    st.rerun()
