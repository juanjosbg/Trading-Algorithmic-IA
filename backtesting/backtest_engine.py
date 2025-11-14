import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

from feature_engineering import add_features
from ml_model import load_model
from backtesting.metrics import max_drawdown, sharpe_ratio  # ← usamos tu metrics.py


class BacktestEngine:

    def __init__(self, symbol, start, end, initial_capital=10000):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_capital = initial_capital

        self.df = None
        self.model = None
        self.feature_cols = None

    # ==========================
    # 1. Cargar datos
    # ==========================
    def load_data(self):
        print(f"\n📥 Descargando datos de {self.symbol} desde {self.start} hasta {self.end}...\n")

        df = yf.download(self.symbol, start=self.start, end=self.end, interval="1d")
        df = add_features(df)
        df.dropna(inplace=True)

        self.df = df

    # ==========================
    # 2. Cargar modelo
    # ==========================
    def load_trading_model(self):
        model, feature_cols = load_model(self.symbol)
        self.model = model
        self.feature_cols = feature_cols

    # ==========================
    # 3. Ejecutar señales ML
    # ==========================
    def run_model_predictions(self):
        print("🤖 Generando señales con el modelo ML...")

        self.df["signal"] = self.model.predict(self.df[self.feature_cols])

        # 1 = LONG, 0 = fuera del mercado
        self.df["market_return"] = self.df["Close"].pct_change()

        # Shift para evitar lookahead bias
        self.df["strategy_return"] = self.df["signal"].shift(1) * self.df["market_return"]

    # ==========================
    # 4. Calcular curva de capital
    # ==========================
    def compute_equity_curve(self):
        print("📈 Calculando curva de capital...")
        self.df["equity_curve"] = (1 + self.df["strategy_return"]).cumprod() * self.initial_capital

    # ==========================
    # 5. Métricas usando metrics.py
    # ==========================
    def compute_metrics(self):
        df = self.df.dropna()

        final_equity = df["equity_curve"].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        returns = df["strategy_return"].dropna().values

        sharpe = sharpe_ratio(returns)
        mdd = max_drawdown(df["equity_curve"].values)
        win_rate = float((returns > 0).mean()) if returns.size > 0 else 0.0

        metrics = {
            "Final Equity": float(final_equity),
            "Total Return %": float(total_return * 100),
            "Win Rate %": float(win_rate * 100),
            "Sharpe Ratio": float(sharpe),
            "Max Drawdown %": float(mdd * 100),  
        }

        print("\n📊 RESULTADOS DEL BACKTEST")
        for k, v in metrics.items():
            print(f"➡ {k}: {v}")

        return metrics

    # ==========================
    # 6. Ejecutar Backtest completo
    # ==========================
    def run(self):
        print("\n🚀 Ejecutando Backtest...\n")

        self.load_data()
        self.load_trading_model()
        self.run_model_predictions()
        self.compute_equity_curve()

        return self.compute_metrics()


if __name__ == "__main__":
    bt = BacktestEngine(
        symbol="AAPL",
        start="2020-01-01",
        end="2024-12-31"
    )
    results = bt.run()
