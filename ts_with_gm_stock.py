# %% GET DATA FROM YAHOO FINANCE AND VISUALIZE

import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt


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


# %% SEASONAL MANN-KENDALL TEST


from statsmodels.graphics.tsaplots import plot_acf
import pymannkendall as mk


# Apply Seasonal Mann-Kendall test and print the result
result = mk.seasonal_test(gm_data["Close"], period=7)
print(result)


# Plot the autocorrelation function for 200 lagging time periods
fig, ax = plt.subplots(figsize=(12, 6))
acf = plot_acf(gm_data["Close"], lags=50, ax=ax)
for line in acf.lines:
    line.set_linewidth(0.5)
plt.show()
