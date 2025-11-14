import yfinance as yf
import pandas as pd

# 1. CONFIGURACIÓN BÁSICA
SYMBOL = "AAPL"   # Puedes cambiarlo a "BTC-USD", "MSFT", etc.
PERIOD = "6mo"    # 6 meses de datos
INTERVAL = "1d"   # velas diarias

SHORT_WINDOW = 20  # media móvil rápida
LONG_WINDOW = 50   # media móvil lenta


def get_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Descarga datos históricos usando yfinance y normaliza columnas."""
    data = yf.download(symbol, period=period, interval=interval)

    # Normalizar columnas si vienen en MultiIndex, como ('Close', 'AAPL')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    else:
        data.columns = [str(c) for c in data.columns]

    return data.dropna()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega medias móviles simples (SMA) al DataFrame."""
    df = df.copy()
    df["SMA_SHORT"] = df["Close"].rolling(window=SHORT_WINDOW).mean()
    df["SMA_LONG"] = df["Close"].rolling(window=LONG_WINDOW).mean()
    return df


def generate_signal(row) -> str:
    """
    Regla simple:
    - BUY: cuando SMA_SHORT > SMA_LONG
    - SELL: cuando SMA_SHORT < SMA_LONG
    - HOLD: en cualquier otro caso
    """
    sma_short = float(row["SMA_SHORT"])
    sma_long = float(row["SMA_LONG"])

    # Si por alguna razón no hay datos suficientes:
    if pd.isna(sma_short) or pd.isna(sma_long):
        return "NO_SIGNAL"

    if sma_short > sma_long:
        return "BUY"
    elif sma_short < sma_long:
        return "SELL"
    else:
        return "HOLD"


def main():
    print(f"Descargando datos de {SYMBOL}...")
    df = get_data(SYMBOL, PERIOD, INTERVAL)

    if df.empty:
        print("❌ No se pudieron obtener datos.")
        return

    df = add_indicators(df)
    df = df.dropna()

    if df.empty:
        print("❌ No hay suficientes datos para calcular las medias móviles.")
        return

    last_row = df.iloc[-1]

    signal = generate_signal(last_row)

    print("\n=== Último dato disponible ===")
    print(f"Símbolo: {SYMBOL}")
    print(f"Fecha: {df.index[-1]}")
    print(f"Cierre: {float(last_row['Close']):.2f}")
    print(f"SMA {SHORT_WINDOW}: {float(last_row['SMA_SHORT']):.2f}")
    print(f"SMA {LONG_WINDOW}: {float(last_row['SMA_LONG']):.2f}")
    print(f"\n👉 Señal sugerida: {signal}")

    if signal == "BUY":
        print("Interpretación: Tendencia alcista — comprar.")
    elif signal == "SELL":
        print("Interpretación: Tendencia bajista — vender.")
    elif signal == "HOLD":
        print("Interpretación: No hacer nada (mantener).")
    else:
        print("Interpretación: No hay señal clara (datos insuficientes).")


if __name__ == "__main__":
    main()
