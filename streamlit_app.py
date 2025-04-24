import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
import ccxt

st.set_page_config(page_title="🛡️ Crypto Alerts App (No Push)", layout="centered")
st_autorefresh(interval=15000, key="refresh")

# Price fallback logic
def get_price(symbol):
    symbol = symbol.upper()
    try:
        res = requests.get(f"https://cex.io/api/ticker/{symbol}/USD", timeout=5)
        if res.status_code == 200:
            return float(res.json()["last"]), "CEX.IO"
    except:
        pass
    try:
        ids = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "XRP": "ripple", "DOGE": "dogecoin", "LTC": "litecoin", "ADA": "cardano"
        }
        cg_id = ids.get(symbol)
        if cg_id:
            cg = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usdt", timeout=5)
            return float(cg.json()[cg_id]["usdt"]), "CoinGecko"
    except:
        pass
    try:
        kraken = ccxt.kraken()
        ticker = kraken.fetch_ticker(f"{symbol}/USDT")
        return ticker["last"], "Kraken"
    except:
        return None, "Unavailable"

st.title("📡 Crypto Price Alerts (Multi-Exchange)")
default_coins = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LTC", "ADA"]
custom = st.text_input("➕ Add Coin", key="add")
coin_list = default_coins + ([custom.upper()] if custom and custom.upper() not in default_coins else [])
symbol = st.selectbox("Select Coin", coin_list, index=0, key="coin_pick")
forecast_minutes = st.slider("Forecast Minutes Ahead", 1, 30, 5)
alert_threshold = st.slider("Alert if Move > USDT", 0.5, 10.0, 2.0)

# Generate simulated price history
np.random.seed(0)
df = pd.DataFrame({
    "timestamp": pd.date_range(end=datetime.now(), periods=100, freq="5T"),
    "close": 100 + np.cumsum(np.random.randn(100))
})
df["returns"] = df["close"].pct_change()
df["sma"] = df["close"].rolling(10).mean()
df["ema"] = df["close"].ewm(span=10).mean()
df["rsi"] = df["returns"].rolling(14).mean() / (df["returns"].rolling(14).std() + 1e-8)
df["future"] = df["close"].shift(-forecast_minutes)
df.dropna(inplace=True)

features = ["close", "returns", "sma", "ema", "rsi"]
X = df[features]
y = df["future"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_scaled[:-1], y[:-1])
pred = model.predict([X_scaled[-1]])[0]

price, source = get_price(symbol)
delta = pred - price if price else 0
direction = "UP 📈" if delta > 0 else "DOWN 📉"
alert = abs(delta) >= alert_threshold if price else False

if price:
    st.metric(f"{symbol}/USDT (via {source})", f"{price:.2f} USDT")
    st.metric("Predicted Price", f"{pred:.2f} USDT")
    st.metric("Expected Move", f"{direction} {abs(delta):.2f} USDT")
    if alert:
        st.success("🔔 ALERT TRIGGERED!")
    else:
        st.info("No alert triggered.")
else:
    st.error("⚠️ Could not fetch price from any exchange.")

# Chart
with st.expander("📊 View Chart"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sma"], name="SMA", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema"], name="EMA", line=dict(dash="dot")))
    st.plotly_chart(fig, use_container_width=True)

# Gold ticker + News
st.markdown("### 🟡 Gold Price Ticker (USD)")
components.html("""
    <iframe src="https://goldbroker.com/widget/live-price/gold/1?currency=USD" 
            width="100%" height="60" frameborder="0" scrolling="no"></iframe>
""", height=60)

st.markdown("### 🗞️ Market News Feed")
components.html("""
    <iframe src="https://rss.app/embed/v1/wall/your_widget_id" 
            width="100%" height="600" frameborder="0" scrolling="no"></iframe>
""", height=600)