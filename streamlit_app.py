import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go
import ccxt
import feedparser
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Set page
st.set_page_config(page_title="📈 Crypto Monitor & Predictor", layout="wide", initial_sidebar_state="expanded")
st.title("📈 AI Crypto Monitor + Strategy Signals")
st.markdown("---")

# Manual Refresh
if st.button("🔄 Refresh Now"):
    st.experimental_rerun()

st.caption(f"Last Refresh: {datetime.now().strftime('%H:%M:%S')}")

# === Utilities ===
def format_price(price):
    return f"${price:,.2f}"

def format_percent(change):
    return f"{change:+.2f}%"

def get_percent_color(change):
    if change > 0:
        return "green"
    elif change < 0:
        return "red"
    else:
        return "gray"

def create_price_history_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name=symbol))
    fig.update_layout(title=f"{symbol} Price History", xaxis_title="Time", yaxis_title="Price (USDT)")
    return fig

# === MultiExchange Monitor ===
class MultiExchangeCryptoMonitor:
    def __init__(self):
        self.exchanges = {
            "pionex": "https://api.pionex.com/api/v1/market/ticker",
            "coingecko": "https://api.coingecko.com/api/v3/simple/price",
        }
        self.kraken = ccxt.kraken()
        self.symbols = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

    def get_price(self, symbol):
        try:
            res = requests.get(self.exchanges["pionex"], params={"symbol": symbol.lower()+"_usdt"}, timeout=5)
            if res.ok:
                return float(res.json()["data"]["price"]), "Pionex"
        except:
            pass
        try:
            ids = {"BTC":"bitcoin","ETH":"ethereum","SOL":"solana","XRP":"ripple","DOGE":"dogecoin"}
            res = requests.get(self.exchanges["coingecko"], params={"ids": ids.get(symbol, symbol.lower()), "vs_currencies":"usdt"}, timeout=5)
            if res.ok:
                return float(list(res.json().values())[0]["usdt"]), "CoinGecko"
        except:
            pass
        try:
            self.kraken.load_markets()
            price = self.kraken.fetch_ticker(f"{symbol}/USDT")["last"]
            return price, "Kraken"
        except:
            return None, "Unavailable"

    def get_ohlcv(self, symbol, timeframe="5m", limit=300):
        try:
            self.kraken.load_markets()
            pair = f"{symbol}/USDT" if f"{symbol}/USDT" in self.kraken.symbols else f"{symbol}/USD"
            data = self.kraken.fetch_ohlcv(pair, timeframe, limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except:
            return pd.DataFrame()

    def get_ticker(self, symbol):
        price, source = self.get_price(symbol)
        if price:
            return {
                "last": price,
                "bid": price * 0.999,
                "ask": price * 1.001,
                "volume": np.random.randint(1000,50000),
                "change": np.random.uniform(-5,5),
                "timestamp": datetime.now(),
                "exchange": source
            }
        return None

    def predict_price(self, symbol, timeframe, prediction_hours):
        df = self.get_ohlcv(symbol, timeframe)
        if df.empty:
            return None
        df["returns"] = df["close"].pct_change()
        df["sma"] = df["close"].rolling(10).mean()
        df["ema"] = df["close"].ewm(span=10).mean()
        df["rsi"] = df["returns"].rolling(14).mean() / (df["returns"].rolling(14).std() + 1e-8)
        df["future"] = df["close"].shift(-prediction_hours)
        df.dropna(inplace=True)

        features = ["close", "returns", "sma", "ema", "rsi"]
        X = df[features]
        y = df["future"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GridSearchCV(RandomForestRegressor(), {
            "n_estimators": [50,100],
            "max_depth": [5,10,None]
        }, cv=3, n_jobs=-1)
        model.fit(X_scaled[:-1], y[:-1])
        best_model = model.best_estimator_

        pred = best_model.predict([X_scaled[-1]])[0]
        backtest = best_model.predict(X_scaled)
        real = y.values
        mae = np.mean(np.abs(backtest - real))
        acc = 100 - (mae / np.mean(real)) * 100

        return {
            "predicted_price": pred,
            "prediction_df": pd.DataFrame({
                "timestamp": [df.iloc[-1]["timestamp"] + pd.Timedelta(minutes=prediction_hours)],
                "predicted_price": [pred]
            }),
            "percent_change": ((pred - df.iloc[-1]["close"]) / df.iloc[-1]["close"]) * 100,
            "confidence": acc,
            "signal": "BUY" if pred > df.iloc[-1]["close"] else "SELL"
        }

# === App Starts ===
monitor = MultiExchangeCryptoMonitor()

default_coins = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
custom_coin = st.text_input("➕ Add Custom Coin", "")
coin = st.selectbox("Select Cryptocurrency", default_coins + ([custom_coin.upper()] if custom_coin else []))
timeframe = st.selectbox("Select Timeframe", ["1m","5m","15m","1h","4h","1d"], index=3)
prediction_minutes = st.slider("🕒 Prediction Horizon (minutes)", 5, 1440, 30)
alert_threshold = st.slider("🚨 Alert Threshold (USDT)", 0.5, 1000.0, 2.0, step=0.5)

# Price
price, source = monitor.get_price(coin)
st.metric(f"{coin}/USDT Price", format_price(price) if price else "Unavailable", help=f"Source: {source}")

# Prediction
pred_result = monitor.predict_price(coin, timeframe, prediction_minutes)
if pred_result:
    delta = pred_result["predicted_price"] - price
    st.metric("Predicted Price", format_price(pred_result["predicted_price"]))
    st.metric("Expected Move", f"{'📈' if delta>0 else '📉'} {abs(delta):.2f} USDT")
    st.metric("Model Confidence", f"{pred_result['confidence']:.1f}%")
    if price and abs(delta) >= alert_threshold:
        st.success("🚨 ALERT TRIGGERED!")

# Chart
df = monitor.get_ohlcv(coin, timeframe)
if not df.empty:
    df["ema8"] = df["close"].ewm(span=8).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["bullish_cross"] = (df["ema8"].shift(1) < df["ema50"].shift(1)) & (df["ema8"] > df["ema50"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema8"], name="8 EMA", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema50"], name="50 EMA", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema20"], name="20 EMA", line=dict(color="orange")))

    if df["bullish_cross"].any():
        fig.add_trace(go.Scatter(
            x=df[df["bullish_cross"]]["timestamp"],
            y=df[df["bullish_cross"]]["low"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="blue"),
            name="Bullish Crossover"
        ))

    st.plotly_chart(fig, use_container_width=True)

# On-Chain Metrics
st.subheader("📡 On-Chain Metrics")
def fetch_santiment(metric, slug="ethereum"):
    query = {
        "query": f"""
        {{
            getMetric(metric: "{metric}") {{
                timeseriesData(slug: "{slug}", from: "utc_now-7d", to: "utc_now", interval: "1d") {{
                    datetime
                    value
                }}
            }}
        }}
        """
    }
    headers = {
        "Authorization": "Apikey vmjzjb73krrryezu_c7up7chwouddym26",
        "Content-Type": "application/json"
    }
    try:
        res = requests.post("https://api.santiment.net/graphql", headers=headers, json=query)
        data = res.json()
        values = data["data"]["getMetric"]["timeseriesData"]
        if values:
            return values[-1]["value"]
    except:
        return None

col1, col2 = st.columns(2)
with col1:
    mvrv = fetch_santiment("mvrv_usd_z_score")
    if mvrv is not None:
        st.metric("MVRV Z-Score", f"{mvrv:.2f}")
    else:
        st.warning("MVRV data unavailable.")

with col2:
    cap_flow = fetch_santiment("exchange_flow")
    if cap_flow is not None:
        st.metric("Capital Flow", f"{cap_flow:.2f}")
    else:
        st.warning("Capital Flow unavailable.")

# News
st.subheader("📰 Market News")
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
