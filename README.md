# 🤖 Proyecto: IA de Trading Algorítmico – Camino a un Bot Autónomo

## 🧠 Visión del Proyecto

El objetivo final de este proyecto es construir un sistema de **trading algorítmico con IA** que, de manera progresiva, llegue a ser lo más **autónomo** posible:

- Decida **cuándo comprar** y **cuándo vender**.
- Elija **en qué activos** entrar.
- Gestione el capital en varias posiciones.
- Aprenda de los datos históricos y del rendimiento de sus propias decisiones.

⚠️ Importante:  
Aunque el objetivo es acercarse a un sistema “autónomo”, **no existe garantía de ganancias**. Todos los sistemas de trading, incluso con IA, pueden perder dinero. Por eso el proyecto se construye **por fases**, comenzando siempre en modo simulación.

---

## 🧩 Fases del Proyecto

### ✅ Fase 1 – Bot de Señales Simple (`simple_bot.py`)

Esta fase implementa una estrategia clásica basada en **medias móviles simples (SMA)**:

- Se descarga el precio histórico de un activo (ej. `AAPL`).
- Se calculan dos medias móviles:
  - `SMA_SHORT` (rápida, ej. 20 días)
  - `SMA_LONG` (lenta, ej. 50 días)
- Reglas:
  - **BUY** cuando `SMA_SHORT > SMA_LONG`
  - **SELL** cuando `SMA_SHORT < SMA_LONG`
  - **HOLD** en cualquier otro caso

El archivo `simple_bot.py`:

- Descarga datos con `yfinance`.
- Calcula indicadores.
- Muestra una **señal actual** (BUY / SELL / HOLD) y algunos valores clave.

> Esta fase sirve para entender la lógica básica de indicadores y señales, pero todavía no sabemos si la estrategia es buena o mala a largo plazo.

---

### ✅ Fase 2 – Backtesting de la Estrategia (`backtest.py`)

En esta fase se responde la pregunta:

> “¿Qué habría pasado si hubiera usado esta estrategia durante los últimos años?”

El archivo `backtest.py`:

- Descarga datos históricos (ej. 5 años de `AAPL`).
- Aplica la misma lógica de medias móviles (SMA 20/50).
- Calcula:
  - **Curva de capital** de la estrategia.
  - **Curva de capital** de un inversor que solo hace *buy & hold*.
  - **Retorno total (%)** de la estrategia.
  - **Retorno total (%)** del buy & hold.
  - **Número de trades** aproximados ejecutados.
  - **Máximo drawdown** (caída máxima del capital).

Con esto se puede evaluar:

- ¿La estrategia gana más que comprar y mantener?
- ¿Cuánto riesgo (drawdown) asume?
- ¿Vale la pena seguir optimizándola?

---

### ⏳ Fase 3 – Paper Trading / Simulación en Tiempo Real

Una vez el backtest muestra resultados razonables, la siguiente fase será:

- Conectar el sistema a un **bróker con API** (ej. Alpaca para acciones de USA).
- Operar en **modo paper trading**:
  - El bot envía órdenes de compra/venta.
  - El bróker simula las operaciones sin usar dinero real.
- Objetivos:
  - Ver cómo se comporta la estrategia en tiempo real.
  - Detectar errores de lógica, latencia, límites de API, etc.
  - Ajustar parámetros sin riesgo financiero.

Archivos previstos:

- `broker_client.py` → conexión a la API (ej. Alpaca).
- `paper_trading_bot.py` → ejecución de la estrategia en tiempo real.

---

### ⏳ Fase 4 – Optimización y Gestión de Riesgo

Antes de pensar en IA o dinero real, la estrategia debe fortalecerse:

- Probar distintos parámetros (ej. SMA 10/30, 20/50, 50/200).
- Agregar:
  - **Stop Loss** automático.
  - **Take Profit**.
  - Filtros de volatilidad.
  - Filtros de tendencia de largo plazo.
- Incluir reglas de:
  - **Gestión de capital** (no ir all-in en una sola posición).
  - **Diversificación en varias acciones**.

Archivos previstos:

- `optimizer.py` → probar combinaciones de parámetros.
- `portfolio_simulator.py` → simulación de varios activos a la vez.

---

### ⏳ Fase 5 – IA / Machine Learning

Cuando ya exista una base sólida de:

- Datos históricos.
- Resultados de backtesting.
- Experiencias en paper trading.

Se podrán incorporar modelos de **Machine Learning** para que la IA “aprenda” patrones del mercado:

- Modelos posibles:
  - **Random Forest / XGBoost**.
  - **Redes neuronales densas**.
  - **LSTM** para series temporales.
  - **Modelos híbridos** (ML + reglas de indicadores).
- Objetivos:
  - Predecir probabilidad de subida/bajada.
  - Reforzar las decisiones de compra/venta.
  - Ajustar dinámicamente parámetros según el rendimiento.

Archivos previstos:

- `feature_engineering.py` → construcción de variables de entrada (features).
- `ml_model.py` → entrenamiento, validación y predicción.
- `ml_trading_bot.py` → integración del modelo con la lógica de trading.

---

### ⏳ Fase 6 – Trading Real (Con Mucha Precaución)

Solo cuando:

- El backtesting sea consistentemente positivo.
- El paper trading funcione bien.
- La IA muestre resultados estables.

Se podría evaluar conectar el sistema a una cuenta real:

- Modo real del bróker (ej. Alpaca live, u otro).
- Capital controlado y limitado.
- Límites estrictos de riesgo.

⚠️ En esta fase:
- No hay garantías de ganancias.
- Siempre existe riesgo de pérdida total del capital asignado.
- El sistema debe monitorearse y revisarse constantemente.

---

## 🧰 Tecnologías Usadas

- **Lenguaje:** Python 3.x
- **Librerías actuales:**
  - `yfinance` → descarga de datos de mercado.
  - `pandas` → análisis de datos.
  - `numpy` → cálculos numéricos.
- **Librerías futuras:**
  - `scikit-learn`, `xgboost` → modelos tradicionales de ML.
  - `tensorflow` o `pytorch` → redes neuronales.
  - SDK del bróker (ej. `alpaca-trade-api`) → conexión a mercado.

---

## 📂 Estructura Actual del Proyecto

```txt
Trading-Algorithmic-IA/
│
├── venv/                 # Entorno virtual de Python
├── simple_bot.py         # Bot de señales simple con medias móviles
├── backtest.py           # Backtesting de la estrategia de medias móviles
└── README.md             # Documentación del proyecto
