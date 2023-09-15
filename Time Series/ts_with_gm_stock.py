# %% GET DATA FROM YAHOO FINANCE AND VISUALIZE

import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# Fetch GM stock data for the last 365 days
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)
gm_data = yf.download("GM", start=start_date, end=end_date).dropna()


# Plot GM stock data
gm_data["Close"].plot(figsize=(10, 6))
plt.title("GM Stock Close Price for Last 100 Days")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

#################################################################################################
# %% SEASONAL MANN-KENDALL TEST


from statsmodels.graphics.tsaplots import plot_acf
import pymannkendall as mk


# Apply Seasonal Mann-Kendall test and print the result
result = mk.seasonal_test(gm_data["Close"], period=12)
print(result)


# Plot the autocorrelation function for 200 lagging time periods
fig, ax = plt.subplots(figsize=(12, 6))
acf = plot_acf(gm_data["Close"], lags=50, ax=ax)
for line in acf.lines:
    line.set_linewidth(0.5)
plt.show()

#################################################################################################
# %% SEASONAL DECOMPOSITION

# Libraries
from statsmodels.tsa.seasonal import seasonal_decompose


# Seasonal decomposition and drop NaN from the results
result = seasonal_decompose(gm_data["Close"], model="additive", period=12)
gm_data["Trend"] = result.trend.dropna()
gm_data["Seasonal"] = result.seasonal.dropna()
gm_data["Residual"] = result.resid.dropna()
gm_data.dropna(inplace=True)  # Drop rows with NaN values

# Visualize decompositions
plt.figure(figsize=(12, 8))
# Observed plot
plt.subplot(411)
plt.plot(gm_data["Close"], label="Observed")
plt.legend(loc="upper left")
# Trend plot
plt.subplot(412)
plt.plot(result.trend, label="Trend")
plt.legend(loc="upper left")
# Seasonal plot
plt.subplot(413)
plt.plot(result.seasonal, label="Seasonal")
plt.legend(loc="upper left")
# Residual plot
plt.subplot(414)
plt.plot(result.resid, label="Residual")
plt.legend(loc="upper left")

# Display the plots
plt.show()

#################################################################################################
# %% SARIMA TEST

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


# Define train and test data
train_size = int(len(gm_data) * 0.7)
train = gm_data["Close"][:train_size]
test = gm_data["Close"][train_size:]

# Fit the SARIMA model | order(p,d,q)
sarima_model = sm.tsa.statespace.SARIMAX(
    train, trend="n", order=(1, 1, 0), seasonal_order=(0, 1, 0, 60)
)

# Print SARIMAX results
results = sarima_model.fit()
print(results.summary())
