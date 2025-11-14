# optimizer.py
"""
Prueba varias combinaciones de medias móviles para encontrar
cuáles parámetros han funcionado mejor en el pasado.
"""

import yfinance as yf
import pandas as pd

SYMBOL = "AAPL"
PERIOD = "5y"
INTERVAL = "1d"
INITIAL_CAPITAL = 10_000


def get_data():
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL)
    return df.dropna()


def run_strategy(df: pd.DataFrame, short_window: int, long_window: int) -> float:
    df = df.copy()
    df["SMA_SHORT"] = df["Close"].rolling(window=short_window).mean()
    df["SMA_LONG"] = df["Close"].rolling(window=long_window).mean()
    df["signal"] = 0
    df.loc[df["SMA_SHORT"] > df["SMA_LONG"], "signal"] = 1
    df["position"] = df["signal"].shift(1).fillna(0)
    df["market_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["market_return"] * df["position"]
    df = df.dropna()

    equity = (1 + df["strategy_return"]).cumprod() * INITIAL_CAPITAL
    final_equity = equity.iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100
    return total_return


def main():
    df = get_data()
    results = []

    for short in [5, 10, 20]:
        for long in [30, 50, 100, 200]:
            if short >= long:
                continue
            r = run_strategy(df, short, long)
            results.append((short, long, r))
            print(f"SMA_SHORT={short}, SMA_LONG={long} → Retorno: {r:.2f}%")

    print("\n=== Mejores parámetros ===")
    results.sort(key=lambda x: x[2], reverse=True)
    for s, l, r in results[:5]:
        print(f"{s}/{l} → {r:.2f}%")


if __name__ == "__main__":
    main()
