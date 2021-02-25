import pandas as pd
from pandas import concat, DataFrame

df = pd.read_csv("BitCoinRaw.csv")
df.set_index("time", inplace=True)
# print(df.tail())

df.drop(columns=["conversionType", "conversionSymbol"], axis=1, inplace=True)

values = DataFrame(df.close.values)
lags = 8
columns = [values]
for i in range(1, (lags + 1)):
    columns.append(values.shift(i))

dt = concat(columns, axis=1)

columns = ["Lag"]
for i in range(1, (lags + 1)):
    columns.append("Lag" + str(i))
dt.columns = columns
dt.index = df.index

finalDataSet = concat([df, dt], axis=1)

finalDataSet.dropna(inplace=True)

finalDataSet["S_10"] = finalDataSet["close"].rolling(window=10).mean()

finalDataSet["Corr"] = (
    finalDataSet["close"].rolling(window=10).corr(finalDataSet["S_10"])
)

finalDataSet["d_20"] = finalDataSet["close"].shift(480)

finalDataSet["5EMA"] = (
    finalDataSet["close"].ewm(span=5, adjust=True, ignore_na=True).mean()
)

finalDataSet["10EMA"] = (
    finalDataSet["close"].ewm(span=10, adjust=True, ignore_na=True).mean()
)

finalDataSet["20EMA"] = (
    finalDataSet["close"].ewm(span=20, adjust=True, ignore_na=True).mean()
)

finalDataSet["mean"] = (finalDataSet["low"] + finalDataSet["high"]) / 2

finalDataSet["returns"] = (
    (finalDataSet["close"] - finalDataSet["open"]) / finalDataSet["open"] * 100.0
)

finalDataSet["volume"] = finalDataSet["volumeto"] - finalDataSet["volumefrom"]

finalDataSet.drop(["volumefrom", "volumeto"], 1, inplace=True)

finalDataSet.dropna(inplace=True)

finalDataSet = finalDataSet.drop(["Lag"], axis=1)

finalDataSet = finalDataSet.astype(float)

finalDataSet = finalDataSet.sort_index(ascending=True)
# dataframe.head(2)

# save data
finalDataSet.to_csv("finalDataSet.csv", header=True)

print(finalDataSet.tail())