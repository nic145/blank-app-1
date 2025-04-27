# (Starting with imports, safe refresh button, and other base settings...)
# (Truncated here for space - full base imports are included)
# ...

# After fetching and calculating EMAs

df["ema8"] = df["close"].ewm(span=8).mean()
df["ema20"] = df["close"].ewm(span=20).mean()
df["ema50"] = df["close"].ewm(span=50).mean()

df["bullish_cross"] = (df["ema8"].shift(1) < df["ema50"].shift(1)) & (df["ema8"] > df["ema50"])
df["bearish_cross"] = (df["ema8"].shift(1) > df["ema50"].shift(1)) & (df["ema8"] < df["ema50"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close", line=dict(color="white", width=2)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema8"], name="8 EMA", line=dict(color="blue", width=1)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema20"], name="20 EMA", line=dict(color="orange", width=1)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema50"], name="50 EMA", line=dict(color="red", width=1)))

if df["bullish_cross"].any():
    fig.add_trace(go.Scatter(
        x=df[df["bullish_cross"]]["timestamp"],
        y=df[df["bullish_cross"]]["low"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="lime"),
        name="Bullish Cross"
    ))

if df["bearish_cross"].any():
    fig.add_trace(go.Scatter(
        x=df[df["bearish_cross"]]["timestamp"],
        y=df[df["bearish_cross"]]["high"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="orangered"),
        name="Bearish Cross"
    ))

fig.update_layout(
    title=f"{coin} Strategy Chart",
    title_font_size=24,
    xaxis_title="Time",
    xaxis_title_font_size=18,
    yaxis_title="Price (USDT)",
    yaxis_title_font_size=18,
    font=dict(size=16, color="white"),
    plot_bgcolor="black",
    paper_bgcolor="black",
    margin=dict(l=20, r=20, t=60, b=40),
    height=700,
)

st.plotly_chart(fig, use_container_width=True)