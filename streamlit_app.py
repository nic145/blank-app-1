import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components

st.set_page_config(
    page_title="📱 CEX.IO Crypto Alerts",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def play_sound():
    sound_url = "https://www.soundjay.com/buttons/sounds/beep-07.mp3"
    st.markdown(f'''
        <audio autoplay>
            <source src="{sound_url}" type="audio/mpeg">
        </audio>
    ''', unsafe_allow_html=True)

def get_cex_price(symbol1, symbol2="USD"):
    url = f"https://cex.io/api/ticker/{symbol1}/{symbol2}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return float(data["last"])
    except Exception as e:
        print(f"Error fetching from CEX.IO: {e}")
        return None

st.title("📡 CEX.IO Crypto Signal Monitor")
st.caption("Live alerts using real-time data from CEX.IO")

default_coins = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LTC"]
custom_coin = st.text_input("➕ Add Custom Coin (e.g. ADA)", key="custom_coin_input")
coin_list = default_coins + ([custom_coin.upper()] if custom_coin and custom_coin.upper() not in default_coins else [])
symbol = st.selectbox("Select Coin", coin_list, index=0, key="coin_select")
forecast_minutes = st.slider("Predict Minutes Ahead", 1, 30, 5, key="forecast_slider")
alert_threshold = st.slider("Trigger Alert If Move > $", 0.5, 10.0, 2.0, step=0.5, key="threshold_slider")

if st.button("🔄 Refresh Prediction"):
    st.session_state["trigger_refresh"] = True

# Fake some simple past data for prediction illustration
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
last_price = get_cex_price(symbol)
delta = pred - last_price if last_price else 0
direction = "UP 📈" if delta > 0 else "DOWN 📉"
alert = abs(delta) >= alert_threshold if last_price else False

if last_price:
    st.metric(label="Current Price", value=f"${last_price:.2f}")
    st.metric(label="Predicted Price", value=f"${pred:.2f}")
    st.metric(label="Expected Move", value=f"{direction} ${abs(delta):.2f}")
    if alert:
        st.success("🔔 ALERT TRIGGERED!")
        play_sound()
    else:
        st.info("No alert triggered.")
else:
    st.error("⚠️ Failed to get real-time price from CEX.IO.")

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