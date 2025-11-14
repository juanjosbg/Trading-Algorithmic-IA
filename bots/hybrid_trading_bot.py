import time
import joblib
import yfinance as yf
import numpy as np
from datetime import datetime

# ==========================
# Configuración del Bot
# ==========================
SYMBOL = "AAPL"
PERIOD = "6mo"
INTERVAL = "1d"
MODEL_PATH = "models/random_forest_aapl.pkl"

MIN_PROBABILITY_TO_BUY = 0.55  # Probabilidad mínima para comprar
MIN_PROBABILITY_TO_SELL = 0.45  # Probabilidad mínima para vender
INTERVAL_SECONDS = 60  # Tiempo entre cada revisión (60s)

# Estado del broker simulado
portfolio = {
    "cash": 5000.0,
    "position": 0,
    "avg_price": 0.0
}

# ==========================
# Funciones auxiliares
# ==========================

def log(msg):
    """Imprimir con timestamp."""
    print(f"\n🕒 {datetime.now()} - {msg}")


def load_model(model_path):
    """Carga el modelo entrenado."""
    model, feature_cols = joblib.load(model_path)
    return model, feature_cols


def compute_sma_signal(df):
    """Calcula una señal simple basada en SMA."""
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    last = df.iloc[-1]

    if last["SMA20"] > last["SMA50"]:
        return "BUY", last["Close"], last["SMA20"], last["SMA50"]

    if last["SMA20"] < last["SMA50"]:
        return "SELL", last["Close"], last["SMA20"], last["SMA50"]

    return "HOLD", last["Close"], last["SMA20"], last["SMA50"]


def compute_features(df):
    """Crea features para el modelo ML."""
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_5"] = df["Close"].pct_change().rolling(5).std()
    df["lag_return_1"] = df["return_1d"].shift(1)
    df = df.dropna()
    return df


def decide_action(ml_prob, sma_signal):
    """Decisión híbrida basada en SMA + IA."""
    if sma_signal == "BUY" and ml_prob >= MIN_PROBABILITY_TO_BUY:
        return "BUY"

    if sma_signal == "SELL" and ml_prob <= MIN_PROBABILITY_TO_SELL:
        return "SELL"

    return "HOLD"


def execute_trade(action, price):
    """Simula una compra o venta."""
    if action == "BUY":
        quantity = int(portfolio["cash"] // price)
        if quantity < 1:
            print("❌ No hay suficiente efectivo para comprar.")
            return

        total_cost = quantity * price
        portfolio["cash"] -= total_cost
        portfolio["position"] += quantity

        # nuevo promedio
        if portfolio["avg_price"] == 0:
            portfolio["avg_price"] = price
        else:
            portfolio["avg_price"] = (portfolio["avg_price"] + price) / 2

        print(f"✅ COMPRA simulada: {quantity} x {SYMBOL} @ {price:.2f}")

    elif action == "SELL":
        if portfolio["position"] == 0:
            print("❌ No tienes acciones para vender.")
            return

        total_value = portfolio["position"] * price
        portfolio["cash"] += total_value
        print(f"🟢 VENTA simulada: {portfolio['position']} x {SYMBOL} @ {price:.2f}")

        portfolio["position"] = 0
        portfolio["avg_price"] = 0


def show_portfolio(price):
    """Imprime el estado del portafolio."""
    total_value = portfolio["cash"] + portfolio["position"] * price

    print("\n=== 🧾 ESTADO DEL BROKER SIMULADO ===")
    print(f"💵 Efectivo: {portfolio['cash']:.2f} USD")
    print(f"📦 {SYMBOL}: {portfolio['position']} @ {portfolio['avg_price']:.2f}")
    print(f"💰 Valor total del portafolio: {total_value:.2f} USD")
    print("======================================\n")


# ==========================
# MAIN LOOP DEL BOT
# ==========================

def main():
    print("🤖 Iniciando Bot Cuantitativo Híbrido (SMA + IA)...")

    model, feature_cols = load_model(MODEL_PATH)

    while True:
        log(f"Revisando {SYMBOL}...")

        df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL)

        sma_signal, price, sma20, sma50 = compute_sma_signal(df)
        print(f"📊 SMA Signal → {sma_signal}")
        print(f"   Close: {price:.2f} | SMA 20: {sma20:.2f} | SMA 50: {sma50:.2f}")

        df_feat = compute_features(df)
        last_row = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)

        ml_prob = model.predict_proba(last_row)[0][1]
        print(f"🤖 IA - Probabilidad de subida: {ml_prob*100:.2f}%")

        action = decide_action(ml_prob, sma_signal)

        if action == "BUY":
            print("🟢 Señal híbrida FUERTE de COMPRA (SMA+IA).")
            execute_trade("BUY", price)

        elif action == "SELL":
            print("🔴 Señal híbrida FUERTE de VENTA (SMA+IA).")
            execute_trade("SELL", price)

        else:
            print("⚪ Señal híbrida débil → HOLD (no se opera).")

        show_portfolio(price)

        print(f"⏳ Esperando {INTERVAL_SECONDS} segundos para la siguiente revisión...")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
