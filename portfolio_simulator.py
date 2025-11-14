# portfolio_simulator.py
"""
Simula un portafolio con múltiples activos usando la estrategia de medias móviles.
"""

import yfinance as yf
import pandas as pd

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
PERIOD = "3y"
INTERVAL = "1d"
SHORT_WINDOW = 20
LONG_WINDOW = 50
INITIAL_CAPITAL = 10_000


def download_data(tickers):
    df = yf.download(tickers, period=PERIOD, interval=INTERVAL)["Close"]
    return df.dropna()


def simulate_portfolio(df: pd.DataFrame):
    if isinstance(df, pd.Series):
        df = df.to_frame()

    capital_per_asset = INITIAL_CAPITAL / df.shape[1]
    equity_total = None

    for symbol in df.columns:
        prices = df[symbol].dropna()
        if len(prices) < LONG_WINDOW + 5:
            continue

        tmp = prices.to_frame(name="Close")
        tmp["SMA_SHORT"] = tmp["Close"].rolling(window=SHORT_WINDOW).mean()
        tmp["SMA_LONG"] = tmp["Close"].rolling(window=LONG_WINDOW).mean()
        tmp["signal"] = 0
        tmp.loc[tmp["SMA_SHORT"] > tmp["SMA_LONG"], "signal"] = 1
        tmp["position"] = tmp["signal"].shift(1).fillna(0)
        tmp["ret"] = tmp["Close"].pct_change()
        tmp["strategy_ret"] = tmp["ret"] * tmp["position"]
        tmp = tmp.dropna()

        tmp["equity"] = (1 + tmp["strategy_ret"]).cumprod() * capital_per_asset

        if equity_total is None:
            equity_total = tmp[["equity"]].rename(columns={"equity": symbol})
        else:
            equity_total = equity_total.join(tmp[["equity"]].rename(columns={"equity": symbol}), how="outer")

    equity_total = equity_total.fillna(method="ffill").dropna()
    equity_total["total_equity"] = equity_total.sum(axis=1)

    print("Valor final del portafolio:", equity_total["total_equity"].iloc[-1])
    print("\nÚltimos valores:")
    print(equity_total.tail())


def main():
    df = download_data(TICKERS)
    simulate_portfolio(df)


if __name__ == "__main__":
    main()
