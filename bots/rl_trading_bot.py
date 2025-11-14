import time
import yfinance as yf
from stable_baselines3 import PPO

model = PPO.load("models/reinforcement_agent.pkl")

def run_rl_bot(symbol="AAPL"):
    print("🤖 Ejecutando RL Trading Bot...")

    df = yf.download(symbol, period="5d", interval="1h")
    obs = df.iloc[-1].values

    action, _ = model.predict(obs, deterministic=True)

    if action == 1:
        print("🟢 RL → BUY")
    elif action == 2:
        print("🔴 RL → SELL")
    else:
        print("🟡 RL → HOLD")

if __name__ == "__main__":
    while True:
        run_rl_bot()
        time.sleep(60)
