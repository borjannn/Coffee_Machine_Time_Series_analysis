import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("coffee_sales.csv", parse_dates=['datetime'])
df = df.set_index('datetime').sort_index()


daily_sales = df['money'].resample('D').sum()

result = adfuller(daily_sales.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

daily_sales.plot(label='Daily Sales', figsize=(10, 5))
daily_sales.rolling(7).mean().plot(label='7-Day Rolling Average of Sales')
daily_sales.rolling(30).mean().plot(label='30-Day Rolling Average of Sales')

plt.title('Daily Coffee Machine Sales and smoothed')
plt.ylabel('Total Sales')

plt.legend()

plt.show()
