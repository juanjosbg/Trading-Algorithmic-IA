import yfinance as yf
import pandas as pd

# CONFIGURACIÓN
SYMBOL = "AAPL"
PERIOD = "6mo"
INTERVAL = "1d"

SHORT_WINDOW = 20
LONG_WINDOW = 50


def get_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(symbol, period=period, interval=interval)
    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA_SHORT"] = df["Close"].rolling(window=SHORT_WINDOW).mean()
    df["SMA_LONG"] = df["Close"].rolling(window=LONG_WINDOW).mean()
    return df


def generate_signal(row):
    sma_short = float(row.get("SMA_SHORT", float("nan")))
    sma_long = float(row.get("SMA_LONG", float("nan")))

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

    if df is None or df.empty:
        print("❌ No se pudieron descargar datos.")
        return

    df = add_indicators(df)
    df = df.dropna()

    if df.empty:
        print("❌ No hay suficientes datos para calcular las medias móviles.")
        return

    last_row = df.iloc[-1].to_dict()

    signal = generate_signal(last_row)

    print("\n=== Último dato disponible ===")
    print(f"Símbolo: {SYMBOL}")
    print(f"Fecha: {df.index[-1]}")
    print(f"Cierre: {float(last_row['Close']):.2f}")
    print(f"SMA {SHORT_WINDOW}: {float(last_row['SMA_SHORT']):.2f}")
    print(f"SMA {LONG_WINDOW}: {float(last_row['SMA_LONG']):.2f}")

    print(f"\n👉 Señal sugerida: {signal}")


if __name__ == "__main__":
    main()
