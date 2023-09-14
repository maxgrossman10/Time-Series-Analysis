# %% GET DATA FROM YAHOO FINANCE AND VISUALIZE


# Libraries
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def get_stock_data(ticker_symbol, days=100):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)

    # Fetch the stock data
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data


def plot_stock_data(data, ticker_symbol):
    data["Close"].plot(figsize=(10, 6))
    plt.title(f"{ticker_symbol} Stock Close Price for Last 100 Days")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    ticker = "GM"
    gm_data = get_stock_data(ticker, 100)
    plot_stock_data(gm_data, ticker)
