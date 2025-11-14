import yfinance as yf
import pandas as pd

# ==========================
# CONFIGURACIÓN DEL BACKTEST
# ==========================
SYMBOL = "AAPL"        # Acción a evaluar
PERIOD = "5y"          # 5 años de datos históricos
INTERVAL = "1d"        # Velas diarias

SHORT_WINDOW = 20      # Media móvil rápida
LONG_WINDOW = 50       # Media móvil lenta

INITIAL_CAPITAL = 10000  # Capital inicial (USD, simulación)


def get_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Descarga datos históricos desde Yahoo Finance."""
    print(f"Descargando datos de {symbol} ({period}, {interval})...")
    df = yf.download(symbol, period=period, interval=interval)
    if df is None or df.empty:
        raise ValueError("No se pudieron descargar datos.")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula medias móviles y genera señales de trading."""
    df = df.copy()

    # Medias móviles (indicadores)
    df["SMA_SHORT"] = df["Close"].rolling(window=SHORT_WINDOW).mean()
    df["SMA_LONG"] = df["Close"].rolling(window=LONG_WINDOW).mean()

    # Señal: 1 = largo (comprado), 0 = fuera del mercado
    df["signal"] = 0
    df.loc[df["SMA_SHORT"] > df["SMA_LONG"], "signal"] = 1

    # Posición efectiva: entramos al día siguiente de la señal
    df["position"] = df["signal"].shift(1).fillna(0)

    # Retornos del mercado y de la estrategia
    df["market_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["market_return"] * df["position"]

    # Limpiar filas iniciales sin datos suficientes
    df = df.dropna()
    return df


def run_backtest(df: pd.DataFrame):
    """Ejecuta el backtest y calcula estadísticas básicas."""
    df = df.copy()

    # Curva de capital de la estrategia
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod() * INITIAL_CAPITAL

    # Benchmark: buy & hold (comprar y mantener)
    df["buy_and_hold"] = (1 + df["market_return"]).cumprod() * INITIAL_CAPITAL

    final_equity = df["equity_curve"].iloc[-1]
    final_bh = df["buy_and_hold"].iloc[-1]

    total_return_strategy = (final_equity / INITIAL_CAPITAL - 1) * 100
    total_return_bh = (final_bh / INITIAL_CAPITAL - 1) * 100

    # Drawdown máximo
    rolling_max = df["equity_curve"].cummax()
    drawdown = df["equity_curve"] / rolling_max - 1
    max_drawdown = drawdown.min() * 100  # en %

    trades = (df["position"].diff().abs() == 1).sum() // 2 

    print("\n===== RESULTADOS DEL BACKTEST =====")
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {PERIOD} | Intervalo: {INTERVAL}")
    print(f"Capital inicial: {INITIAL_CAPITAL:.2f} USD\n")

    print(f"Capital final estrategia: {final_equity:.2f} USD")
    print(f"Retorno total estrategia: {total_return_strategy:.2f}%\n")

    print(f"Capital final Buy & Hold: {final_bh:.2f} USD")
    print(f"Retorno total Buy & Hold: {total_return_bh:.2f}%\n")

    print(f"Número aproximado de trades: {trades}")
    print(f"Máximo drawdown (caída máxima): {max_drawdown:.2f}%")
    print("====================================\n")

    # Opcional: mostrar últimas filas
    print("Últimos registros de la curva de capital:")
    print(df[["Close", "equity_curve", "buy_and_hold"]].tail())


def main():
    df = get_data(SYMBOL, PERIOD, INTERVAL)
    df = prepare_data(df)
    run_backtest(df)


if __name__ == "__main__":
    main()
