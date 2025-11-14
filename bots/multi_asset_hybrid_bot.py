# multi_asset_hybrid_bot.py
"""
Bot cuantitativo híbrido multi-activos:
- Usa un solo modelo ML (RandomForest entrenado con AAPL)
- Usa SMA 20/50 por cada símbolo
- Combina SMA + IA para decidir BUY/SELL por activo
- Maneja un portafolio con múltiples posiciones en un broker simulado
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
TICKERS = ["AAPL", "MSFT", "AMZN"]  # puedes agregar más
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


def download_data_symbol(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Descarga datos de UN símbolo y normaliza columnas."""
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
    - señal: "BUY" / "SELL" / "HOLD"
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

    def to_float(x):
        return float(x.item()) if hasattr(x, "item") else float(x)

    price = to_float(last_row["Close"])
    sma_short = to_float(last_row["SMA_SHORT"])
    sma_long = to_float(last_row["SMA_LONG"])

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
    X.columns = feature_cols

    proba_up = model.predict_proba(X)[0][1]
    return proba_up


def main():
    print("🚀 Iniciando Bot Cuantitativo Híbrido Multi-Activos (SMA + IA)...")

    model, feature_cols = load_model()
    broker = SimulatedBroker(cash=10_000.0)

    while True:
        print(f"\n🕒 {datetime.now()} - Revisando portafolio: {', '.join(TICKERS)}")

        prices_for_portfolio = {}

        for symbol in TICKERS:
            print(f"\n=== Analizando {symbol} ===")

            df = download_data_symbol(symbol, PERIOD, INTERVAL)
            if df.empty:
                print(f"❌ No se pudieron obtener datos para {symbol}.")
                continue

            # 1. SMA por símbolo
            sma_signal, price, sma_short, sma_long = compute_sma_signal(df)
            prices_for_portfolio[symbol] = price

            print(f"📊 SMA Signal ({symbol}) → {sma_signal}")
            print(f"   Close: {price:.2f} | SMA {SHORT_WINDOW}: {sma_short:.2f} | SMA {LONG_WINDOW}: {sma_long:.2f}")

            # 2. IA por símbolo (usando mismo modelo entrenado)
            proba_up = compute_ml_signal(df, model, feature_cols)
            if proba_up is None:
                print(f"❌ No se pudo calcular la señal de IA para {symbol}.")
                continue

            print(f"🧠 IA ({symbol}) - Probabilidad de subida: {proba_up:.2%}")

            # 3. Lógica híbrida por símbolo

            # presupuesto: fracción del cash disponible repartida entre los activos
            # por ejemplo: 20% del cash / número de símbolos
            allocation_fraction = 0.2
            budget = (broker.cash * allocation_fraction) / len(TICKERS)

            if sma_signal == "BUY" and proba_up > 0.55:
                qty = int(budget // price)
                if qty > 0:
                    print(f"✅ Señal híbrida FUERTE de COMPRA en {symbol} (SMA+IA).")
                    broker.buy(symbol, qty, price)
                else:
                    print(f"⚠️ No alcanza presupuesto para comprar al menos 1 acción de {symbol}.")

            elif sma_signal == "SELL" and proba_up < 0.45:
                if symbol in broker.positions:
                    qty = broker.positions[symbol].qty
                    print(f"✅ Señal híbrida FUERTE de VENTA en {symbol} (SMA+IA).")
                    broker.sell(symbol, qty, price)
                else:
                    print(f"ℹ️ Señal de venta en {symbol}, pero no hay posición abierta.")
            else:
                print(f"🤝 Señales en desacuerdo o débiles en {symbol} → HOLD (no se opera).")

        # Mostrar estado global del portafolio
        broker.print_status(prices_for_portfolio)

        print(f"⏳ Esperando {INTERVAL_SECONDS} segundos para la siguiente revisión...\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
