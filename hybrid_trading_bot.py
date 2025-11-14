# hybrid_trading_bot.py
"""
Bot cuantitativo híbrido:
Combina señales de análisis técnico (SMA 20/50)
+ señales de IA (RandomForest) para decidir BUY/SELL
usando un broker simulado.
"""

import time
from datetime import datetime

import yfinance as yf
import pandas as pd
import joblib

from broker_client import SimulatedBroker
from feature_engineering import add_basic_features

# =========================
# CONFIGURACIÓN DEL BOT
# =========================
SYMBOL = "AAPL"
PERIOD = "1y"
INTERVAL = "1d"

SHORT_WINDOW = 20    # SMA rápida
LONG_WINDOW = 50     # SMA lenta

INTERVAL_SECONDS = 60  # cada cuántos segundos revisa el mercado

MODEL_PATH = "models/random_forest_aapl.pkl"


def load_model():
    """Carga el modelo entrenado y las columnas de features."""
    model, feature_cols = joblib.load(MODEL_PATH)
    return model, feature_cols


def download_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Descarga datos del símbolo y normaliza columnas."""
    df = yf.download(symbol, period=period, interval=interval)
    df = df.dropna()

    # Normalizar columnas si vienen como MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    return df


def compute_sma_signal(df: pd.DataFrame) -> tuple[str, float, float, float]:
    """
    Calcula SMA 20/50 y devuelve:
    - señal SMA: "BUY" / "SELL" / "HOLD"
    - precio de cierre
    - SMA_SHORT
    - SMA_LONG
    """
    df = df.copy()
    df["SMA_SHORT"] = df["Close"].rolling(window=SHORT_WINDOW).mean()
    df["SMA_LONG"] = df["Close"].rolling(window=LONG_WINDOW).mean()
    df = df.dropna()

    if df.empty:
        return "NO_SIGNAL", float("nan"), float("nan"), float("nan")

    last_row = df.iloc[-1]

    close_val = last_row["Close"]
    sma_short_val = last_row["SMA_SHORT"]
    sma_long_val = last_row["SMA_LONG"]

    # Conversión segura a float
    def to_float(x):
        return float(x.item()) if hasattr(x, "item") else float(x)

    price = to_float(close_val)
    sma_short = to_float(sma_short_val)
    sma_long = to_float(sma_long_val)

    if pd.isna(sma_short) or pd.isna(sma_long):
        return "NO_SIGNAL", price, sma_short, sma_long

    if sma_short > sma_long:
        signal = "BUY"
    elif sma_short < sma_long:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, price, sma_short, sma_long


def compute_ml_signal(df: pd.DataFrame, model, feature_cols):
    """
    Aplica feature engineering y devuelve:
    - probabilidad de subida (float entre 0 y 1)
    """
    df_feat = add_basic_features(df)
    df_feat = df_feat.dropna()
    if df_feat.empty:
        return None

    last_row = df_feat.iloc[-1]

    # Construimos X con las mismas columnas que usó el modelo
    X = last_row[feature_cols].to_frame().T
    X.columns = feature_cols  # aseguramos orden/nombres

    proba_up = model.predict_proba(X)[0][1]  # probabilidad de que suba
    return proba_up


def main():
    print("🚀 Iniciando Bot Cuantitativo Híbrido (SMA + IA)...")

    model, feature_cols = load_model()
    broker = SimulatedBroker(cash=5_000.0)

    while True:
        print(f"\n🕒 {datetime.now()} - Revisando {SYMBOL}...")

        # 1. Descargamos datos recientes
        df = download_data(SYMBOL, PERIOD, INTERVAL)
        if df.empty:
            print("❌ No se pudieron obtener datos de mercado.")
            time.sleep(INTERVAL_SECONDS)
            continue

        # 2. Señal por SMA
        sma_signal, price, sma_short, sma_long = compute_sma_signal(df)
        print(f"📊 SMA Signal → {sma_signal}")
        print(f"   Close: {price:.2f} | SMA {SHORT_WINDOW}: {sma_short:.2f} | SMA {LONG_WINDOW}: {sma_long:.2f}")

        # 3. Señal por IA
        proba_up = compute_ml_signal(df, model, feature_cols)
        if proba_up is None:
            print("❌ No se pudo calcular la señal de IA (datos insuficientes).")
            time.sleep(INTERVAL_SECONDS)
            continue

        print(f"🧠 IA - Probabilidad de subida: {proba_up:.2%}")

        prices = {SYMBOL: price}

        # ===============================
        # 4. LÓGICA HÍBRIDA DE DECISIÓN
        # ===============================

        # Zonas:
        # PRO-ALCISTA: SMA = BUY y IA > 0.55
        # PRO-BAJISTA: SMA = SELL y IA < 0.45
        # En desacuerdo → HOLD

        if sma_signal == "BUY" and proba_up > 0.55:
            budget = 700
            qty = int(budget // price)
            if qty > 0:
                print("✅ Señal híbrida FUERTE de COMPRA (SMA+IA).")
                broker.buy(SYMBOL, qty, price)
            else:
                print("⚠️ No alcanza el presupuesto para comprar al menos 1 acción.")

        elif sma_signal == "SELL" and proba_up < 0.45:
            if SYMBOL in broker.positions:
                qty = broker.positions[SYMBOL].qty
                print("✅ Señal híbrida FUERTE de VENTA (SMA+IA).")
                broker.sell(SYMBOL, qty, price)
            else:
                print("ℹ️ Señal de venta, pero no hay posición abierta.")

        else:
            print("🤝 Señales en desacuerdo o débiles → HOLD (no se opera).")

        broker.print_status(prices)

        print(f"⏳ Esperando {INTERVAL_SECONDS} segundos para la siguiente revisión...\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
