import pandas as pd
from datetime import datetime

df = pd.read_csv("headlines.csv")
df = df.dropna(subset=["Headline"])  # Remove rows with empty headlines

# Convert the "Date" column to datetime objects
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

publication_date_range = df.groupby("Publication")["Date"].agg(["min", "max"])
min_date, max_date = publication_date_range.iloc[0]

# Iterate through the date ranges to find the overlap
for _, row in publication_date_range.iterrows():
    min_date = max(min_date, row["min"])
    max_date = min(max_date, row["max"])

df = df[df["Date"] >= min_date]
df = df[df["Date"] <= max_date]

print(df.head())

import matplotlib.pyplot as plt

# Count the number of headlines per day for each publication
df_count = df.groupby(["Date", "Publication"]).size().unstack().fillna(0)

# Plotting the line chart
plt.figure(figsize=(10, 6))
for publication in df_count.columns:
    plt.plot(df_count.index, df_count[publication], label=publication)

plt.xlabel("Date")
plt.ylabel("Number of Headlines")
plt.title("Number of Headlines Over Time by Publication")
plt.legend()
plt.show()
