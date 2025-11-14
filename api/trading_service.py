# api/trading_service.py
import yfinance as yf
import numpy as np
import pandas as pd

from feature_engineering import add_features
from ml_model import load_model

ASSETS = ["AAPL", "MSFT", "AMZN"]  # puedes cambiar esta lista


def get_symbol_signal(symbol: str):
    df = yf.download(symbol, period="6mo", interval="1d")
    df = add_features(df)
    df = df.dropna()

    if df.empty:
        return None

    # Cargamos modelo (usando tu función existente)
    model, feature_cols = load_model(symbol)

    last_row = df.iloc[-1]
    X = last_row[feature_cols].values.reshape(1, -1)

    prob_up = model.predict_proba(X)[0][1]  
    pred = model.predict(X)[0]  

    # Señal SMA simple
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    last = df.iloc[-1]

    sma_signal = "HOLD"
    if last["SMA20"] > last["SMA50"]:
        sma_signal = "BUY"
    elif last["SMA20"] < last["SMA50"]:
        sma_signal = "SELL"

    price = float(last["Close"])

    return {
        "symbol": symbol,
        "price": price,
        "ml_prob_up": float(prob_up),
        "ml_signal": "BUY" if pred == 1 else "FLAT",
        "sma_signal": sma_signal,
    }


def position_size(capital: float, price: float, confidence: float, max_risk_pct=0.02):
    """
    capital: capital total disponible
    confidence: probabilidad [0,1]
    max_risk_pct: % máximo a arriesgar del capital por trade
    """
    if price <= 0:
        return 0

    risk_capital = capital * max_risk_pct * confidence  # más confianza → más tamaño
    qty = int(risk_capital // price)

    return max(qty, 0)


def get_recommendations(capital: float):
    """
    Devuelve recomendaciones por activo según tu capital.
    """
    signals = []

    for symbol in ASSETS:
        info = get_symbol_signal(symbol)
        if not info:
            continue

        # Combinar SMA + ML
        strong_buy = (
            info["sma_signal"] == "BUY" and info["ml_prob_up"] >= 0.55
        )
        strong_sell = (
            info["sma_signal"] == "SELL" and info["ml_prob_up"] <= 0.45
        )

        action = "HOLD"
        if strong_buy:
            action = "BUY"
        elif strong_sell:
            action = "SELL"

        suggested_qty = 0
        if action == "BUY":
            suggested_qty = position_size(
                capital=capital,
                price=info["price"],
                confidence=info["ml_prob_up"],
            )

        signals.append({
            "symbol": info["symbol"],
            "price": info["price"],
            "ml_prob_up": info["ml_prob_up"],
            "ml_signal": info["ml_signal"],
            "sma_signal": info["sma_signal"],
            "action": action,
            "suggested_qty": suggested_qty,
        })

    return signals
