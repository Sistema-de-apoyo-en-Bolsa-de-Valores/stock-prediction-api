# app\infrastructure\yahoo_finance\yahoo_finance.py

import yfinance as yf
import pandas as pd

def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data[['Close']].dropna()