# Algorithmic Trading with AI

A research project that evolves a trading bot step by step: from a classic moving-average strategy, to a **machine learning** model, to a **reinforcement learning** agent. Everything runs in **simulation and paper trading** first, and signals are exposed through a **FastAPI** service.

> ⚠️ Educational project. It is not financial advice and there is no guarantee of profit.

---

## Project phases

| Phase | File | Approach |
|---|---|---|
| 1. Simple signals | `simple_bot.py` | Fast and slow moving-average crossover (SMA 20 / 50) |
| 2. Machine learning | `ml_model.py`, `ml_trading_bot.py` | Random Forest trained on engineered features to predict price direction |
| 3. Reinforcement learning | `rl_agent.py`, `bots/rl_trading_bot.py` | PPO agent (Stable-Baselines3) in a custom Gymnasium environment |
| 4. Hybrid bot | `bots/hybrid_trading_bot.py`, `bots/multi_asset_hybrid_bot.py` | Combines ML and RL signals across several assets |
| 5. Operations | `paper_trading_bot.py`, `bots/retraining_scheduler.py` | Paper trading and scheduled model retraining |

## Features

- **Feature engineering** on historical market data from Yahoo Finance
- **Backtesting engine** with performance metrics (returns, drawdown, win rate)
- **Portfolio simulator** and parameter **optimizer**
- **REST API** (FastAPI) that returns buy / sell / hold recommendations per asset
- Pre-trained model included (`models/random_forest_aapl.pkl`)

## Tech stack

Python · pandas · NumPy · scikit-learn · Stable-Baselines3 (PPO) · Gymnasium · yfinance · FastAPI

## Project structure

```
├── api/           FastAPI server, schemas and trading service
├── backtesting/   Backtest engine and metrics
├── bots/          RL, hybrid and multi-asset bots, retraining scheduler
├── models/        Saved models
├── feature_engineering.py
├── ml_model.py
├── rl_agent.py
└── paper_trading_bot.py
```

## Getting started

```bash
git clone https://github.com/juanjosbg/Trading-Algorithmic-IA.git
cd Trading-Algorithmic-IA
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn yfinance gymnasium stable-baselines3 fastapi uvicorn

# Run a backtest
python backtest.py

# Start the API
uvicorn api.main:app --reload
```

Then open `http://localhost:8000/docs` to try the endpoints.

## Roadmap

- [ ] React dashboard to visualize signals and backtest results
- [ ] Add news sentiment as a feature using an LLM
- [ ] Dockerize the API

## Author

**Juan José Borrero** · [Portfolio](https://profile-juanjose.vercel.app/) · [GitHub](https://github.com/juanjosbg)
