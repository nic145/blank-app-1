import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh
import ccxt
import feedparser
from bs4 import BeautifulSoup

st.set_page_config(page_title="📱 Mobile AI Crypto Alerts", layout="centered")
st_autorefresh(interval=30000, key="refresh_30s")

st.markdown("<h1 style='text-align:center;'>📱 AI Crypto Alerts (With On-Chain Data)</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

colA, colB = st.columns(2)
with colA:
    st.button("🔮 Predict Now", use_container_width=True)
with colB:
    refresh_news = st.button("🔄 Refresh News", use_container_width=True)

# Coin + time inputs
default_coins = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
custom_coin = st.text_input("➕ Custom Coin", "")
coin = st.selectbox("Select Coin", default_coins + ([custom_coin.upper()] if custom_coin else []))
forecast_minutes = st.slider("🕒 Predict (minutes ahead)", 5, 1440, 30, step=5)
alert_threshold = st.slider("🚨 Alert Threshold (USDT)", 0.5, 10.0, 2.0)

# Fetch price
def get_pionex_price(symbol):
    url = "https://api.pionex.com/api/v1/market/ticker"
    params = {"symbol": symbol.lower() + "_usdt"}
    try:
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
    st.warning("⚠️ Simulated data used.")
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=300, freq="5T"),
        "close": 100 + np.cumsum(np.random.randn(300))
    })

# Indicators
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

mae = np.mean(np.abs(backtest - real))
acc = 100 - (mae / np.mean(real)) * 100

price, source = get_price(coin)
delta = pred - price if price else 0
direction = "📈 Up" if delta > 0 else "📉 Down"
alert = abs(delta) >= alert_threshold if price else False

# Metrics
st.metric(f"{coin}/USDT Price", f"{price:.4f}" if price else "Unavailable")
st.metric("Predicted", f"{pred:.4f}")
st.metric("Expected Move", f"{direction} {abs(delta):.4f}")
st.metric("Accuracy Estimate", f"{acc:.2f}%")
if alert:
    st.success("🚨 ALERT TRIGGERED!")
else:
    st.info("No alert triggered.")

# Charts
with st.expander("📊 Price Chart"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sma"], name="SMA", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema"], name="EMA", line=dict(dash="dot")))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🧪 Backtesting"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=real, name="Actual"))
    fig.add_trace(go.Scatter(y=backtest, name="Predicted"))
    st.plotly_chart(fig, use_container_width=True)

# News section using feedparser
st.markdown("### 🗞️ Market News (Cointelegraph)")
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
    st.warning("No news available at the moment.")

# On-chain metrics from Santiment
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
        st.metric("MVRV Z-Score", f"{latest['value']:.2f}", delta=None)
    else:
        st.warning("MVRV Z-Score unavailable")
with col2:
    cap_flow = fetch_santiment_metric("exchange_flow")
    if cap_flow:
        latest = cap_flow[-1]
        st.metric("Capital Flow", f"{latest['value']:.2f}")
    else:
        st.warning("Capital Flow unavailable")