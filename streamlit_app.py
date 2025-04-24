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

# Optional: Pushover for mobile push alerts
def send_push_notification(message, user_key, app_token):
    url = "https://api.pushover.net/1/messages.json"
    payload = {
        "token": app_token,
        "user": user_key,
        "message": message,
        "title": "Crypto Alert",
        "priority": 1
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
    except Exception as e:
        st.warning(f"Push notification failed: {e}")

st.set_page_config(page_title="🛡️ Multi-Source Crypto Alerts", layout="centered")
st_autorefresh(interval=15000, key="refresh")

# Sidebar Settings
st.sidebar.title("🔔 Alert Settings")
pushover_user = st.sidebar.text_input("Pushover User Key")
pushover_token = st.sidebar.text_input("Pushover App Token", type="password")

# Price fallback system
def get_price(symbol):
    symbol_upper = symbol.upper()
    try:
        # Try CEX.IO
        url_cex = f"https://cex.io/api/ticker/{symbol_upper}/USD"
        response = requests.get(url_cex, timeout=5)
        if response.status_code == 200:
            return float(response.json()["last"]), "CEX.IO"
    except:
        pass

    try:
        # Try CoinGecko
        cg_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "XRP": "ripple",
            "DOGE": "dogecoin",
            "LTC": "litecoin",
            "ADA": "cardano"
        }
        coingecko_id = cg_map.get(symbol_upper)
        if coingecko_id:
            cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usdt"
            response = requests.get(cg_url, timeout=5)
            if response.status_code == 200:
                return float(response.json()[coingecko_id]["usdt"]), "CoinGecko"
    except:
        pass

    try:
        # Try Kraken via ccxt
        exchange = ccxt.kraken()
        ticker = exchange.fetch_ticker(f"{symbol_upper}/USDT")
        return ticker["last"], "Kraken"
    except:
        return None, "Unavailable"

# Feature section
st.title("📡 Crypto Price Alerts with Fallbacks & Push")
st.caption("Live predictions with multi-exchange reliability and mobile alerts.")

default_coins = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LTC", "ADA"]
custom_coin = st.text_input("➕ Add Custom Coin", key="custom_coin_input")
coin_list = default_coins + ([custom_coin.upper()] if custom_coin and custom_coin.upper() not in default_coins else [])
symbol = st.selectbox("Select Coin", coin_list, index=0, key="coin_select")
forecast_minutes = st.slider("Predict Minutes Ahead", 1, 30, 5)
alert_threshold = st.slider("Alert Threshold (USDT)", 0.5, 10.0, 2.0, step=0.5)

# Simulate historical data
np.random.seed(0)
price_series = pd.Series(100 + np.cumsum(np.random.randn(100)))
timestamp_series = pd.date_range(end=datetime.now(), periods=100, freq="5T")
df = pd.DataFrame({"timestamp": timestamp_series, "close": price_series})
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

# Price and alert logic
last_price, source = get_price(symbol)
delta = pred - last_price if last_price else 0
direction = "UP 📈" if delta > 0 else "DOWN 📉"
alert = abs(delta) >= alert_threshold if last_price else False

if last_price:
    st.metric(label=f"{symbol}/USDT (via {source})", value=f"{last_price:.2f} USDT")
    st.metric(label="Predicted Price", value=f"{pred:.2f} USDT")
    st.metric(label="Expected Move", value=f"{direction} {abs(delta):.2f} USDT")
    if alert:
        st.success("🔔 ALERT TRIGGERED!")
        st.write(f"Exchange: {source}")
        if pushover_user and pushover_token:
            send_push_notification(
                f"{symbol}/USDT is expected to move {direction} by {abs(delta):.2f} USDT",
                pushover_user, pushover_token
            )
    else:
        st.info("No alert triggered.")
else:
    st.error("⚠️ Could not retrieve live price from any exchange.")

# Chart
with st.expander("📊 View Chart"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sma"], name="SMA", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema"], name="EMA", line=dict(dash="dot")))
    st.plotly_chart(fig, use_container_width=True)

# Gold ticker and news feed
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