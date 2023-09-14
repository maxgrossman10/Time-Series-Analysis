# %% GET DATA FROM YAHOO FINANCE AND VISUALIZE

import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pymannkendall as mk

# Fetch GM stock data for the last 100 days
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=100)
gm_data = yf.download("GM", start=start_date, end=end_date)

# Plot GM stock data
gm_data["Close"].plot(figsize=(10, 6))
plt.title("GM Stock Close Price for Last 100 Days")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()
