import gymnasium as gym
import gym_anytrading
from operator import itemgetter

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

#Quant Finance
from finta import TA
import quantstats as qs

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
import yfinance as yfin
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle

from good_performing_stocks import return_stock_list

yfin.pdr_override()

end_date =  datetime.now().date() - timedelta(days=30)
start_date = end_date - timedelta(days=395)

def get_profits(ticker_name):
    prices = pdr.get_data_yahoo(ticker_name, start=start_date, end=end_date)
    data = prices
    data['return'] = np.log(data['Close'] / data['Close'].shift(1))
    env_maker = lambda: gym.make('stocks-v0', df=data, frame_bound = (5,100), window_size=2)
    env = DummyVecEnv([env_maker])
    model = A2C('MlpPolicy', env, verbose=0)
    # model.learn(total_timesteps=100000)
    # model.save(str(ticker_name))
    model.load('models/' + str(ticker_name))
    env = gym.make('stocks-v0', df=data, frame_bound = (90,100), window_size=2)
    obs, info = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, truncate, done, info = env.step(action)
        if done:
            return info
            break

# profits = get_profits('BAJAJ-AUTO.NS')
# print(profits)

def return_shortlisted_stocks():
    ticker_list = return_stock_list()
    reward_list = {}

    for ticker in ticker_list:
        profits = get_profits(ticker)
        reward_list[ticker] = profits["total_reward"]

    sorted_stock_list=sorted(list(reward_list.items()), key=itemgetter(1), reverse=True)
    final_list = []
    count = 0
    for stocks in sorted_stock_list:
        if count < 10:
            final_list.append(stocks)
            count += 1
        else:
            break

