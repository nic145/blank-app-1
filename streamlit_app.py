import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import ccxt
import feedparser
from bs4 import BeautifulSoup

# Basic Page Config
st.set_page_config(page_title="📱 Mobile AI Crypto Alerts", layout="centered")

st.markdown("<h1 style='text-align:center;'>📱 AI Crypto + On-Chain Signals</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Manual Refresh Button
if st.button("🔄 Refresh App"):
    st.experimental_rerun()

# Timestamp of last refresh
st.caption(f"Last Refreshed at {datetime.now().strftime('%H:%M:%S')}")

# Inputs
default_coins = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
custom_coin = st.text_input("➕ Custom Coin", "")
coin = st.selectbox("Select Coin", default_coins + ([custom_coin.upper()] if custom_coin else []))
forecast_minutes = st.slider("🕒 Predict (minutes ahead)", 5, 1440, 30, step=5)
alert_threshold = st.slider("🚨 Alert Threshold (USDT)", 0.5, 10.0, 2.0)

# Fetch Price
def get_pionex_price(symbol):
    try:
        url = "https://api.pionex.com/api/v1/market/ticker"
        params = {"symbol": symbol.lower() + "_usdt"}
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        return float(res.json()["data"]["price"]), "Pionex"
    except:
        return None, "Unavailable"

def get_price(symbol):
    price, source = get_pionex_price(symbol)
    if price: return price, source
    ids = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "DOGE": "dogecoin"}
    try:
        cg_id = ids.get(symbol.upper())
        if cg_id:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usdt"
            return float(requests.get(url, timeout=5).json()[cg_id]["usdt"]), "CoinGecko"
    except: pass
    try:
        kraken = ccxt.kraken()
        return kraken.fetch_ticker(f"{symbol}/USDT")["last"], "Kraken"
    except:
        return None, "Unavailable"

# Fetch OHLCV
def get_ohlcv(symbol, timeframe="5m", limit=300):
    try:
        kraken = ccxt.kraken()
        kraken.load_markets()
        pair = f"{symbol}/USDT" if f"{symbol}/USDT" in kraken.symbols else f"{symbol}/USD"
        df = kraken.fetch_ohlcv(pair, timeframe, limit)
        df = pd.DataFrame(df, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except:
        return None

df = get_ohlcv(coin)
if df is None or df.empty:
    st.warning("⚠️ Using simulated data...")
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=300, freq="5T"),
        "close": 100 + np.cumsum(np.random.randn(300))
    })

# Build Indicators
df["returns"] = df["close"].pct_change()
df["sma"] = df["close"].rolling(10).mean()
df["ema"] = df["close"].ewm(span=10).mean()
df["rsi"] = df["returns"].rolling(14).mean() / (df["returns"].rolling(14).std() + 1e-8)
df["future"] = df["close"].shift(-forecast_minutes)
df.dropna(inplace=True)

features = ["close", "returns", "sma", "ema", "rsi"]
X = df[features]
y = df["future"]

if len(X) < 10:
    st.error("Not enough data. Try reducing prediction time.")
    st.stop()

# Train AI Model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gs = GridSearchCV(RandomForestRegressor(), {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 2, 5]
}, cv=3, n_jobs=-1)
gs.fit(X_scaled[:-1], y[:-1])
model = gs.best_estimator_
pred = model.predict([X_scaled[-1]])[0]
backtest = model.predict(X_scaled)
real = y.values

# Accuracy
mae = np.mean(np.abs(backtest - real))
acc = 100 - (mae / np.mean(real)) * 100

# Show Metrics
price, source = get_price(coin)
delta = pred - price if price else 0
direction = "📈 Up" if delta > 0 else "📉 Down"
alert = abs(delta) >= alert_threshold if price else False

st.metric(f"{coin}/USDT Price", f"{price:.4f}" if price else "Unavailable", help=f"Source: {source}")
st.metric("Predicted", f"{pred:.4f}")
st.metric("Expected Move", f"{direction} {abs(delta):.4f}")
st.metric("Accuracy Estimate", f"{acc:.2f}%")
if alert:
    st.success("🚨 ALERT TRIGGERED!")

# === EMA Strategy Chart ===
st.subheader("📈 EMA Strategy")

df["ema8"] = df["close"].ewm(span=8).mean()
df["ema20"] = df["close"].ewm(span=20).mean()
df["ema50"] = df["close"].ewm(span=50).mean()

# Crossovers
df["bullish_cross"] = (df["ema8"].shift(1) < df["ema50"].shift(1)) & (df["ema8"] > df["ema50"])

# Wait for touch of 20 EMA after crossover
waiting_for_touch = False
touch_events = []
for idx in df.index:
    if df.at[idx, "bullish_cross"]:
        waiting_for_touch = True
    if waiting_for_touch and df.at[idx, "low"] <= df.at[idx, "ema20"]:
        touch_events.append(idx)
        waiting_for_touch = False

fig_strategy = go.Figure()
fig_strategy.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close", line=dict(color="white")))
fig_strategy.add_trace(go.Scatter(x=df["timestamp"], y=df["ema8"], name="8 EMA", line=dict(color="blue")))
fig_strategy.add_trace(go.Scatter(x=df["timestamp"], y=df["ema50"], name="50 EMA", line=dict(color="red")))
fig_strategy.add_trace(go.Scatter(x=df["timestamp"], y=df["ema20"], name="20 EMA", line=dict(color="orange")))

if not df[df["bullish_cross"]].empty:
    cross_idx = df[df["bullish_cross"]].index
    fig_strategy.add_trace(go.Scatter(
        x=df.loc[cross_idx, "timestamp"],
        y=df.loc[cross_idx, "low"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="blue"),
        name="Bullish Crossover"
    ))

if touch_events:
    fig_strategy.add_trace(go.Scatter(
        x=df.loc[touch_events, "timestamp"],
        y=df.loc[touch_events, "low"],
        mode="markers",
        marker=dict(symbol="x", size=10, color="orange"),
        name="Touch 20 EMA"
    ))

st.plotly_chart(fig_strategy, use_container_width=True)

# === On-Chain Metrics ===
st.subheader("📡 On-Chain Metrics (Santiment)")

def fetch_santiment_metric(metric_slug, symbol_slug="ethereum"):
    query = {
        "query": f'''
        {{
            getMetric(metric: "{metric_slug}") {{
                timeseriesData(
                    slug: "{symbol_slug}",
                    from: "utc_now-7d",
                    to: "utc_now",
                    interval: "1d"
                ) {{
                    datetime
                    value
                }}
            }}
        }}
        '''
    }
    headers = {
        "Authorization": "Apikey vmjzjb73krrryezu_c7up7chwouddym26",
        "Content-Type": "application/json"
    }
    response = requests.post("https://api.santiment.net/graphql", headers=headers, json=query)
    if response.status_code == 200:
        return response.json()["data"]["getMetric"]["timeseriesData"]
    return []

col1, col2 = st.columns(2)
with col1:
    mvrv = fetch_santiment_metric("mvrv_usd_z_score")
    if mvrv:
        latest = mvrv[-1]
        st.metric("MVRV Z-Score", f"{latest['value']:.2f}")
    else:
        st.warning("MVRV unavailable")
with col2:
    cap_flow = fetch_santiment_metric("exchange_flow")
    if cap_flow:
        latest = cap_flow[-1]
        st.metric("Capital Flow", f"{latest['value']:.2f}")
    else:
        st.warning("Capital Flow unavailable")

# === Market News ===
st.subheader("🗞️ Market News (Cointelegraph)")

feed_url = "https://cointelegraph.com/rss"
feed = feedparser.parse(feed_url)
if feed.entries:
    for entry in feed.entries[:5]:
        st.markdown(f"**{entry.title}**")
        st.caption(entry.published)
        summary = BeautifulSoup(entry.summary, "html.parser").get_text()
        st.write(summary)
        st.markdown(f"[Read more]({entry.link})")
        st.markdown("---")
else:
    st.warning("No news available.")