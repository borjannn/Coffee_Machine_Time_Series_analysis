import pandas as pd
import matplotlib as mpl

mpl.use("TkAgg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("coffee_sales.csv", parse_dates=["datetime"])
df = df.set_index("datetime").sort_index()

daily_sales = df["money"].resample("D").sum()

df["day_of_week"] = df.index.day_name()
weekday_sales = df.groupby("day_of_week")["money"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

plt.figure(figsize=(8, 4))

weekday_sales.plot(kind='bar', color='tan')
plt.title("Average Sales by Day of Week")
plt.ylabel("Average Sales")
plt.xlabel("Day of Week")
plt.tight_layout()
plt.show()

df["hour"] = df.index.hour
hourly_sales = df.groupby("hour")["money"].mean()

hourly_sales.plot(marker="o")
plt.title("Average Sales by Hour of Day")
plt.ylabel("Average Sales")
plt.xlabel("Hour of Day")
plt.tight_layout()
plt.show()

plot_acf(df["money"].resample("h").sum(), lags=48)
plt.title("Hourly Autocorrelation (Detect Daily Seasonality)")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(daily_sales.dropna(), lags=30, ax=axes[0])
axes[0].set_title("ACF (Autocorrelation)")

plot_pacf(daily_sales.dropna(), lags=30, ax=axes[1], method="ywm")
axes[1].set_title("PACF (Partial Autocorrelation)")

plt.tight_layout()
plt.show()

monthly_sales = df["money"].resample("ME").sum()
monthly_sales.plot(marker="o", title="Monthly Coffee Sales Totals", figsize=(8, 4))
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()


