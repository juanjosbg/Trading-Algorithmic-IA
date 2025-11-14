# ml_trading_bot.py
"""
Bot de trading que usa un modelo ML (RandomForest) para decidir buy/sell
en un broker simulado.
"""

import yfinance as yf
import pandas as pd
import joblib
import time
from datetime import datetime

from broker_client import SimulatedBroker
from feature_engineering import add_basic_features

SYMBOL = "AAPL"
INTERVAL_SECONDS = 60
MODEL_PATH = "models/random_forest_aapl.pkl"


def load_model():
    model, feature_cols = joblib.load(MODEL_PATH)
    return model, feature_cols


def get_latest_features(symbol: str, period: str = "1y", interval: str = "1d", feature_cols=None):
    df = yf.download(symbol, period=period, interval=interval)
    df = df.dropna()
    df = add_basic_features(df)
    df = df.dropna()

    # Última fila
    last_row = df.iloc[-1]

    # Convertir Close de manera segura
    price = float(last_row["Close"].item()) if hasattr(last_row["Close"], "item") else float(last_row["Close"])

    # Solo features que el modelo espera
    X = last_row[feature_cols].to_frame().T
    X.columns = feature_cols  # corregir nombres

    return X, price


def main():
    model, feature_cols = load_model()
    broker = SimulatedBroker(cash=5_000.0)

    while True:
        print(f"\n🕒 {datetime.now()} - ML Bot revisando {SYMBOL}...")

        X, price = get_latest_features(SYMBOL, feature_cols=feature_cols)

        # Probabilidad de que suba mañana
        proba_up = model.predict_proba(X)[0][1]

        print(f"📈 Precio actual: {price:.2f}")
        print(f"🧠 Probabilidad de subida: {proba_up:.2%}")

        prices = {SYMBOL: price}

        # Reglas mejoradas – más activas!
        if proba_up > 0.55:
            budget = 500
            qty = int(budget // price)
            if qty > 0:
                broker.buy(SYMBOL, qty, price)

        elif proba_up < 0.45:
            if SYMBOL in broker.positions:
                qty = broker.positions[SYMBOL].qty
                broker.sell(SYMBOL, qty, price)

        broker.print_status(prices)

        print(f"⏳ Esperando {INTERVAL_SECONDS} segundos...\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
