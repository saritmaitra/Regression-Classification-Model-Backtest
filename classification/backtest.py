!pip install pyforest
from pyforest import *
import datetime, pickle, copy, warnings
!pip install cryptocompare
import cryptocompare
from time import time
from datetime import datetime
from pandas import DataFrame, concat
from sklearn import metrics, preprocessing
from math import sqrt
!pip install pyfolio
import pyfolio as pf
!pip install backtrader
import backtrader as bt
from backtrader.feeds import PandasData
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys, logging, json, pprint, requests
from google.colab import files

...
apiKey = "insert api key"
url = "https://min-api.cryptocompare.com/data/histohour"

# BTC data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000
}

result = requests.get(url, params=payload).json()

btc1 = DataFrame(result['Data'])

# 2nd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (btc1.time.head(1)-1) # result['TimeFrom'] #
}

result = requests.get(url, params=payload).json()

btc2 = DataFrame(result['Data'])

# 3rd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (btc2.time.head(1)-1)
}

result = requests.get(url, params=payload).json()

btc3 = DataFrame(result['Data'])

# 4th 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (btc3.time.head(1)-1)
}

result = requests.get(url, params=payload).json()

btc4 = DataFrame(result['Data'])

# combining BTC dataframe
com1 = btc2.append(btc1)
com2 = btc3.append(com1)
btc = btc4.append(com2)

# ETH DATA
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000
}

result = requests.get(url, params=payload).json()
eth1 = DataFrame(result['Data'])

# 2nd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (eth1.time.head(1)-1)
}

result = requests.get(url, params=payload).json()
eth2 = DataFrame(result['Data'])

# 3rd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (eth2.time.head(1)-1)
}

result = requests.get(url, params=payload).json()
eth3 = DataFrame(result['Data'])

# 4th ETH 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (eth3.time.head(1)-1)
}

result = requests.get(url, params=payload).json()

eth4 = DataFrame(result['Data'])

# combining BTC dataframe
com1 = eth2.append(eth1)
com2 = eth3.append(com1)
eth = eth4.append(com2)

# LTC data
payload = {
    "api_key": apiKey,
    "fsym": "LTC",
    "tsym": "USD",
    "limit": 2000
}
result = requests.get(url, params=payload).json()
ltc1 = DataFrame(result['Data'])

# 2nd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "LTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (ltc1.time.head(1)-1)
}

result = requests.get(url, params=payload).json()
ltc2 = DataFrame(result['Data'])

# 3rd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "LTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (ltc2.time.head(1)-1)
}

result = requests.get(url, params=payload).json()
ltc3 = DataFrame(result['Data'])

# 4th ETH 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (ltc3.time.head(1)-1)
}

result = requests.get(url, params=payload).json()

ltc4 = DataFrame(result['Data'])

# combining dataframe
com1 = ltc2.append(ltc1)
com2 = ltc3.append(com1)
ltc = ltc4.append(com2)


# --Data Selection
from pandas import DataFrame, concat

df = DataFrame({'ETH': eth.close})
dataframe = concat([btc, df], axis=1)
dataframe.drop(columns = ['conversionType','conversionSymbol'], axis=1, inplace=True)

values = DataFrame(btc.close.values)
lags = 8
columns = [values]
for i in range(1,(lags + 1)):
    columns.append(values.shift(i))
dt = concat(columns, axis=1)
columns = ['Lag']
for i in range(1,(lags + 1)):
    columns.append('Lag' + str(i))
dt.columns = columns
dt.index = dataframe.index

dataframe = concat([dataframe, dt], axis=1)

# converting to pandas dataframe
dataframe['time'] = pd.to_datetime(dataframe['time'],unit='s')
dataframe.set_index('time',inplace=True)

dataframe.loc[:,'S_10'] = dataframe.loc[:,'close'].rolling(window=10).mean()
dataframe.loc[:,'Corr'] = dataframe.loc[:,'close'].rolling(window=10).corr(dataframe['S_10'])
dataframe.loc[:,'5EMA'] = (dataframe.loc[:,'close'].ewm(span=5,adjust=True,ignore_na=True).mean())
dataframe.loc[:,'10EMA'] = (dataframe.loc[:,'close'].ewm(span=10,adjust=True,ignore_na=True).mean())
dataframe.loc[:,'20EMA'] = (dataframe.loc[:,'close'].ewm(span=20,adjust=True,ignore_na=True).mean())
dataframe.loc[:,'mean'] = (dataframe.loc[:,'low'] + dataframe['high'])/2
dataframe.loc[:,'returns'] = (dataframe.loc[:,'close'] - dataframe['open']) / dataframe['open'] * 100.0
dataframe.loc[:,'volume'] = dataframe.loc[:,'volumeto'] - dataframe.loc[:,'volumefrom']
dataframe.drop(['volumefrom', 'volumeto'], 1, inplace=True)

dataframe.loc[:,'day_of_week'] = dataframe.index.dayofweek
dataframe.loc[:,'day_of_month'] = dataframe.index.day
# dataframe.loc[:,'quarter'] = dataframe.index.quarter
dataframe.loc[:,'month'] = dataframe.index.month
# dataframe.loc[:,'year'] = dataframe.index.year

dataframe.dropna(inplace=True)

dataframe = dataframe.drop(['Lag'], axis=1)

dataframe = dataframe.sort_index(ascending=True)


...
# If any of the values of percentage returns equal zero, setting them to
# a small number (stops issues with LDA model)

for i,x in enumerate(dataframe.loc[:,"returns"]):
    if (abs(x) < 0.0001):
        dataframe.loc[:,"returns"][i] = 0.0001
            
# Create the lagged percentage returns columns

for i in range(0, lags):
    dataframe["Lag%s" % str(i+1)] = dataframe["Lag%s" % str(i+1)].pct_change()*100.0

# Create the "Direction" column (+1 or -1) indicating an up/down period

dataframe.loc[:,"Direction"] = np.sign(dataframe.loc[:,"returns"])

dataframe.to_csv('btc_data.csv') # saving data

...
df = pd.read_csv("btc_data.csv")
df.set_index('time', inplace=True)
df.sort_index(ascending=True, inplace=True)
df.index = pd.to_datetime(df.index)

df.dropna(inplace=True)
MLDataFrame = df.copy()

...
X = np.array(df.drop(['open', 'high',
                      'low', 'close', 'Direction',
                      'Lag6', 'Lag7', 'Lag8',
                       'returns'], axis=1))

X = X.astype(float)
X = preprocessing.scale(X)
y = np.array(df['Direction'])

MLDataFrame = MLDataFrame[['open', 'high', 'low', 'close', 'volume', 'Direction']]

...
lda = LinearDiscriminantAnalysis().fit(X,y)
MLDataFrame.loc[:,'PredictedSignal'] = lda.predict(X)

"""
Instead of predict, we are considering probability of occurrence
"""
# Probability of occurrence (last 24 hours prediction)
MLDataFrame.loc[:, 'positions'] = lda.predict_proba(X)

# introducing differenced to restrict buy/sell
# MLDataFrame.loc[:,'positions'] = MLDataFrame['PredictedSignal'].diff().fillna(0)

print('Number of trades (buy) = ', (MLDataFrame['positions']> 0.60).sum())
print('Number of trades (sell) = ', (MLDataFrame['positions']> 0.80).sum())
print()

%matplotlib inline
MLDataFrame.positions.value_counts()

buys = MLDataFrame.loc[MLDataFrame["positions"] > 0.60] 
sells = MLDataFrame.loc[MLDataFrame["positions"] > 0.80]

"""
- Buy when model prediction > 60 % confident
- Sell when model prediction > 70 % confident
"""
# Plot
fig = plt.figure(figsize=(15, 6))
plt.plot(MLDataFrame.index[-100:], MLDataFrame["close"][-100:])

plt.plot(
    buys.index[-100:],
    MLDataFrame.loc[buys.index]["close"][-100:],
    "v",
    markersize=10,
    color="red",
    label="Buy",
)
plt.plot(
    sells.index[-100:],
    MLDataFrame.loc[sells.index]["close"][-100:],
    "^",
    markersize=10,
    color="g",
    label="Sell",
)
plt.title("Trading Strategy")
plt.show()


...
prices = MLDataFrame.copy()

prices.drop(['Direction'], 1, inplace=True)

OHLCV = ['open', 'high', 'low', 'close', 'volume']

...
# class to define the columns we will provide

class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """
    cols = OHLCV + ['positions']

    # create lines
    lines = tuple(cols)

    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())

...
# define backtesting strategy class

class ML_TestStrategy(bt.Strategy):
    # params = (
    #     ('period', 1000),
    #     )
 

   
    def __init__(self):
      '''Initializes logger and variables required for the strategy implementation.'''
      # initialize logger for log function (set to critical to prevent any unwanted autologs, not using log objects because only care about logging one thing)
      
      for handler in logging.root.handlers[:]:
          logging.root.removeHandler(handler)

      logging.basicConfig(format='%(message)s', level=logging.CRITICAL, handlers=[
          logging.FileHandler("LOG.log"),
          logging.StreamHandler()
          ]
          )
      
      self.startCash = self.broker.getvalue()
      
      # keep track of open, close prices and predicted value in the series
      
      self.data_positions = self.datas[0].positions
      self.data_open = self.datas[0].open
      self.data_close = self.datas[0].close
        
      # keep track of pending orders/buy price/buy commission
      
      self.order = None
      self.price = None
      self.stop_price = None
      self.comm = None

    # logging function

    """
    log function allows us to pass in data via the txt variable that we want to output to the screen. 
    It will attempt to grab datetime values from the most recent data point,if available, and log it to the screen.
    """
    
    def log(self, txt, doprint=True):
        '''Logging function'''
        dt = self.datetime.datetime(ago=0)
        time = self.data.datetime.time(0)
        print('%s, %s' % (dt.isoformat(), txt))
           
 

    def notify_order(self, order):

      """
      Run on every next iteration, logs the order execution status whenever an order is filled or rejected, 
      setting the order parameter back to None if the order is filled or cancelled to denote that there are no more pending orders.
      """

      if order.status in [order.Submitted, order.Accepted]:
          # order already submitted/accepted - no action required

          return

      # report executed order
      if order.status == order.Completed:
          if order.isbuy():
              self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
              # self.log(f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
              # )
              # self.price = order.executed.price
              # self.comm = order.executed.comm

            
          else:
              self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
              # self.log(f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
              # )
              # self.price = order.executed.price
          self.bar_executed = len(self)

      # report failed order
      elif order.status in [order.Canceled, order.Margin, order.Rejected]:
          self.log('Order rejected/margin')

      """
      When system receives a buy or sell signal, we can instruct it to create an order. 
      However, that order won’t be executed until the next bar is called, at whatever price that may be.
      """
      
      
      # set no pending order
      self.order = None

      """
      The next item we will overwrite is the notify_order function. 
      This is where everything related to trade orders gets processed.
      this will log when an order gets executed, and at what price. 
      This will also provide notification in case an order didn’t go through.
      """

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

        """
        As we have predicted the market direction on the day’s closing price, hence we will use cheat_on_open=True 
        when creating the bt.Cerebro object. 
        This means the number of shares we want to buy will be based on day t+1’s open price. 
        As a result, we also define the next_open method instead of next within the Strategy class.
        """

    def next_open(self):
      # Check if we are in the market
        if not self.position:
            if self.data_positions>0.60:

                # calculate the max number of shares ('all-in')
                size = int(self.broker.getcash() / self.datas[0].open)
               
                # buy order
                self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                self.buy(size=size)
        else:
            if self.data_positions>0.80:

                # sell order
                self.log(f'SELL CREATED --- Size: {self.position.size}')
                self.sell(size=self.position.size)

"""
Code commentary:
 - The function __init__ tracks open, close, predicted, and pending orders.
 - The function notify_order tracks the order status.
 - The function notify_trade is triggered if the order is complete and logs profit and loss for the trade.
 - The function next_open checks the available cash and calculates the maximum number of shares that can be bought. 
It places the buy order if we don’t hold any position and the predicted value is > zero. 
Else, it places the sell order if the predicted value < zero.
"""

...
# instantiate SignalData class

data = SignalData(dataname=prices)

def runstrat():

    # Variable for our starting cash

    startCash = 100000.0

    # instantiate Cerebro, add strategy, data, initial cash, commission and pyfolio for performance analysis

    SARIT = bt.Cerebro(stdstats = False, cheat_on_open=True, maxcpus=1)
    SARIT.addstrategy(ML_TestStrategy)

    SARIT.adddata(data)

    # Set our desired cash start

    SARIT.broker.setcash(startCash)
    SARIT.broker.setcommission(commission=0.001)
    SARIT.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    startPortfolioValue = SARIT.broker.getvalue()
    print('Starting Portfolio Value:', startPortfolioValue)
    print()
    
     
    SARIT.run()

    endPortfolioValue = SARIT.broker.getvalue()
    print()
    print(f'Final Portfolio Value: {endPortfolioValue:.2f}')
        
    pnl = endPortfolioValue - startPortfolioValue
    print()
    print(f'PnL: {pnl:.2f}') 

    SARIT.plot(style = 'candlestick')[0][0].savefig('samplefigure.png', dpi=300)
    files.download('samplefigure.png')


if __name__ == '__main__':
    runstrat()

