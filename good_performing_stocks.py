# IMPORT LIBRARIES
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
import yfinance as yfin
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle

yfin.pdr_override()

series_ticker = pickle.load(open('ticker_data.p', 'rb'))

def perform_calculations(ticker, start_date, end_date, time_period):
    
    pct_return_after_period = []
    buy_dates = []
    try:
        prices = pdr.get_data_yahoo(ticker, start=start_date, end=end_date).Close
        prices.index = [d.date() for d in prices.index]
    except (RemoteDataError, KeyError):
        return None, -np.inf, np.inf, None
    
    

    for buy_date, buy_price in prices.items():
        sell_date = buy_date + timedelta(weeks=time_period)
        try:
            sell_price = prices[prices.index == sell_date].iloc[0]
        except IndexError:
            continue 
        
        # Compute returns 
        pct_return = (sell_price - buy_price) / buy_price * 100
        pct_return_after_period.append(pct_return)
        buy_dates.append(buy_date)
    
    return prices, np.mean(pct_return_after_period), np.std(pct_return_after_period), [buy_dates, pct_return_after_period]

# Set parameters
end_date =  datetime.now().date()
start_date = end_date - timedelta(days=300)

time_period = 4 # week
target_returns = 3
acceptable_deviation = 7

def return_stock_list():
    stock_list = []
    ticker_list = []
    for i in range(len(series_ticker)):
        ticker = series_ticker.index[i]
        name = series_ticker.iloc[i].values[0]
        prices, avg_returns, std_devation_returns, returns = perform_calculations(ticker, start_date, end_date, time_period)
        if avg_returns >= target_returns and std_devation_returns <= acceptable_deviation:
            # plot_stock_trends_and_returns(prices, name, returns)
            ticker_list.append(ticker)
            stock_list.append([prices, name, returns, avg_returns, std_devation_returns])
        else:
            continue

    return ticker_list

# stock_list = return_stock_list()
# print("Length of stock list: ",len(stock_list))

# for stocks in stock_list:
#     print("Name: ", stocks[1])
#     print("Avg. Return: ", stocks[3])
