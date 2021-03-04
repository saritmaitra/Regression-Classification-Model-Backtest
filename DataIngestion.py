# machine learning classification
from pyforest import *
import datetime, pickle, copy, warnings
import cryptocompare
import requests
import plotly.express as px
import plotly.graph_objects as go
from time import time
from pandas import DataFrame, concat
from sklearn import metrics
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt
from pandas import DataFrame, concat

...
apiKey = "43b01c420b66888ce4c91b364647600814578c186e8604322152f44c641ebbc1"
url = "https://min-api.cryptocompare.com/data/histohour"

# check if url and API working
check = requests.get(url)
print(check)

# check if url and API working
check = requests.get(url)
print("If Response [200] then working:", check)

# check if url and API working
check = requests.get(url)
print(check.text)
print(check.status_code)
print("If Response [200] then working:", check)

...
# BTC 1st 2000 datapoints
payload = {"api_key": apiKey, "fsym": "BTC", "tsym": "USD", "limit": 2000}

result = requests.get(url, params=payload).json()

BitCoin1 = DataFrame(result["Data"])

# BitCoin1["time"] = pd.to_datetime(BitCoin1["time"], unit="s")

# BitCoin1.set_index("time", inplace=True)

...
# 2nd 2000 datapoints
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (BitCoin1.time.head(1) - 1),
}

result = requests.get(url, params=payload).json()

BitCoin2 = DataFrame(result["Data"])

# BitCoin2["time"] = pd.to_datetime(BitCoin2["time"], unit="s")

# BitCoin2.set_index("time", inplace=True)

...
# 3rd 2000 datapoints
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (BitCoin2.time.head(1) - 1),
}

result = requests.get(url, params=payload).json()

BitCoin3 = DataFrame(result["Data"])

# BitCoin3["time"] = pd.to_datetime(BitCoin3["time"], unit="s")

# BitCoin3.set_index("time", inplace=True)

...
# 4th 2000 datapoints
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (BitCoin3.time.head(1) - 1),
}

result = requests.get(url, params=payload).json()

BitCoin4 = DataFrame(result["Data"])

# BitCoin4["time"] = pd.to_datetime(BitCoin4["time"], unit="s")

# BitCoin4.set_index("time", inplace=True)

...
# combining all bitcoin data (8000 data points)
combineData1 = BitCoin2.append(BitCoin1)

combineData2 = BitCoin3.append(combineData1)

BitCoin = BitCoin4.append(combineData2)  # final BitCoin dataset

# converting to pandas dataframe
BitCoin["time"] = pd.to_datetime(BitCoin["time"], unit="s")
BitCoin.set_index("time", inplace=True)

print(BitCoin.tail(2))
# saving btc data set
BitCoin.to_csv("BitCoinRaw.csv")
