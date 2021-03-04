import pandas as pd
from pandas import concat, DataFrame
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoLars
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import sklearn.externals
import joblib

...
finalDataSet = pd.read_csv("finalDataSet.csv")
finalDataSet.set_index("time", inplace=True)
# print(df.tail())

...
foreCastColumn = "close"  # creating label

foreCastOut = int(12)  # prediction for next 12 hrs

finalDataSet["label"] = finalDataSet[foreCastColumn].shift(-foreCastOut)

...
X = np.array(finalDataSet.drop(["label"], axis=1))

# normalize data
X = preprocessing.scale(X)

XforeCastOut = X[-foreCastOut:]

X = X[:-foreCastOut]

finalDataSet.dropna(inplace=True)

y = np.array(finalDataSet["label"])

...
# Split the data into train and test data set
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

...
# regression model
Model = LassoLars(alpha=0.01).fit(X, y).fit(X_train, y_train)

...
# cross validated accucary on train set
scores = cross_val_score(Model, X_train, y_train, cv=tscv)

print("Training Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Intercept:", Model.intercept_)
print("Slope:", Model.coef_[0])

...
# prediction on training
trainPredict = Model.predict(X_train)
r_squared = r2_score(y_train, trainPredict)
mae = np.mean(abs(trainPredict - y_train))
rmse = np.sqrt(np.mean((trainPredict - y_train) ** 2))
rae = np.mean(abs(trainPredict - y_train)) / np.mean(abs(y_train - np.mean(y_train)))
rse = np.mean((trainPredict - y_train) ** 2) / np.mean(
    (y_train - np.mean(y_train)) ** 2
)
sumOfDf = DataFrame(
    index=[
        "R-squared",
        "Mean Absolute Error",
        "Root Mean Squared Error",
        "Relative Absolute Error",
        "Relative Squared Error",
    ]
)
sumOfDf["Training metrics"] = [r_squared, mae, rmse, rae, rse]

# prediction of test
testPredict = Model.predict(X_test)
r_squared = r2_score(y_test, testPredict)
mae = np.mean(abs(testPredict - y_test))
rmse = np.sqrt(np.mean((testPredict - y_test) ** 2))
rae = np.mean(abs(testPredict - y_test)) / np.mean(abs(y_test - np.mean(y_test)))
rse = np.mean((testPredict - y_test) ** 2) / np.mean((y_test - np.mean(y_test)) ** 2)

sumOfDf["Validation metrics"] = [r_squared, mae, rmse, rae, rse]
sumOfDf = sumOfDf.round(decimals=3)

print(sumOfDf)  # accuracy check

...
# Save model to file in the current working directory
fileName = "ElasticModel.pkl"
joblib.dump(Model, fileName)

# Load from file
ElasticModel = joblib.load(fileName)
# ElasticModel.predict(X_test)
# print(r2_score(y_test, ElasticModel.predict(X_test)))

# forecast future 12 hrs values
foreCastFutureValues = DataFrame(ElasticModel.predict(XforeCastOut))
# print(foreCastFutureValues)

...
# assigning names to columns
foreCastFutureValues.rename(columns={0: "Forecast"}, inplace=True)

newDataframe = finalDataSet.tail(foreCastOut)

newDataframe.reset_index(inplace=True)

newDataframe = newDataframe.append(
    DataFrame(
        {
            "time": pd.date_range(
                start=newDataframe.time.iloc[-1],
                periods=(len(newDataframe) + 1),
                freq="H",
                closed="right",
            )
        }
    )
)

newDataframe.set_index("time", inplace=True)

newDataframe = newDataframe.tail(foreCastOut)

foreCastFutureValues.index = newDataframe.index

print("12 hours forecast (hourly):")
foreCastFutureValues.reset_index(inplace=True)

print(foreCastFutureValues)
