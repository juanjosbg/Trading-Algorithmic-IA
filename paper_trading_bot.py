"""
Bot de "paper trading" simulado.
No usa broker real, sino SimulatedBroker.
"""

import time
from datetime import datetime

import yfinance as yf
import pandas as pd

from broker_client import SimulatedBroker

SYMBOL = "AAPL"
SHORT_WINDOW = 20
LONG_WINDOW = 50
INTERVAL_SECONDS = 30 


def get_latest_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval)
    return df.dropna()


def compute_signal(df: pd.DataFrame) -> str:
    df = df.copy()
    df["SMA_SHORT"] = df["Close"].rolling(window=SHORT_WINDOW).mean()
    df["SMA_LONG"] = df["Close"].rolling(window=LONG_WINDOW).mean()
    df = df.dropna()

    if df.empty:
        return "NO_SIGNAL"

    last = df.iloc[-1]
    if last["SMA_SHORT"] > last["SMA_LONG"]:
        return "BUY"
    elif last["SMA_SHORT"] < last["SMA_LONG"]:
        return "SELL"
    else:
        return "HOLD"


def main():
    broker = SimulatedBroker(cash=5_000.0)

    while True:
        print(f"\n🕒 {datetime.now()} - Revisando señal para {SYMBOL}...")
        df = get_latest_data(SYMBOL)
        if df.empty:
            print("❌ No hay datos, reintentando...")
            time.sleep(INTERVAL_SECONDS)
            continue

        last_price = df["Close"].iloc[-1]
        signal = compute_signal(df)
        print(f"Señal: {signal} | Precio actual: {last_price:.2f}")

        prices = {SYMBOL: float(last_price)}

        if signal == "BUY":
            budget = 500
            qty = budget // last_price
            if qty > 0:
                broker.buy(SYMBOL, qty, last_price)

        elif signal == "SELL":
            if SYMBOL in broker.positions:
                qty = broker.positions[SYMBOL].qty
                broker.sell(SYMBOL, qty, last_price)

        broker.print_status(prices)

        print(f"Esperando {INTERVAL_SECONDS} segundos antes de la próxima revisión...\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
