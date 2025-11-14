import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.current_step = 0
        self.cash = 10000
        self.shares = 0

        self.action_space = gym.spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(df.columns),),
            dtype=np.float32
        )

    def step(self, action):
        price = self.df["Close"].iloc[self.current_step]

        if action == 1 and self.cash >= price:
            self.shares += 1
            self.cash -= price
        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.cash += price

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        portfolio_value = self.cash + self.shares * price
        reward = portfolio_value

        obs = self.df.iloc[self.current_step].values

        return obs, reward, done, False, {}

    def reset(self, seed=None):
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        return self.df.iloc[self.current_step].values, {}

def train_rl_agent(symbol="AAPL"):
    df = yf.download(symbol, period="5y", interval="1d")
    df.dropna(inplace=True)

    env = TradingEnv(df)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    model.save("models/reinforcement_agent.pkl")
    print("🤖 Modelo RL guardado correctamente.")

if __name__ == "__main__":
    train_rl_agent()
