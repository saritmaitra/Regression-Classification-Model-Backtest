# Realtime BitCoin prediction using Azure ML
Here I am developing a machine learning regression algorithm on CryptoCurrency(BitCoin) using ElasticNet (Linear regression with combined L1 and L2 priors as regularizers).
The app predicts the Closing price for BitCoin; the deployment is with real-time inference selecting the model trained on the "Close" column. However, this is a sample project and not to be tried for commercial purpose.

I have used pickling to serilize the model into byte code and store that; while deploying de-serilization is needed.
