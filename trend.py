import pandas as pd
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('coffee_sales.csv', parse_dates=['datetime'])
df = df.set_index('datetime').sort_index()

daily_sales = df['money'].resample('D').sum()

plt.figure(figsize=(10, 5))
plt.plot(daily_sales.index, daily_sales.values, label="Daily Sales", alpha=0.5)
plt.plot(daily_sales.index, daily_sales.rolling(30).mean(), label="30-Day Rolling Average", linewidth=2)
plt.title("Coffee Machine Sales — Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.tight_layout()
plt.show()

decomp = seasonal_decompose(daily_sales, model="additive", period=7)
decomp.plot()
plt.suptitle("Seasonal Decomposition (Trend Focus)")
plt.tight_layout()
plt.show()

x = np.arange(len(daily_sales)).reshape(-1, 1)
y = daily_sales.values
model = LinearRegression().fit(x, y)
trend_line = model.predict(x)

plt.figure(figsize=(10, 5))
plt.plot(daily_sales.index, y, label="Daily Sales", alpha=0.5)
plt.plot(daily_sales.index, trend_line, label="Linear Trend", color="red", linewidth=2)
plt.title("Trend Analysis of Coffee Sales (Linear Fit)")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.tight_layout()
plt.show()

slope = model.coef_[0]
direction = "increasing" if slope > 0 else "decreasing"
print(f"Average trend: sales are {direction} by approximately {abs(slope):.2f} units per day.")

