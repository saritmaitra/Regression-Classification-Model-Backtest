# Realtime BitCoin prediction using Azure ML
It is a sample ML regression algorithm on CryptoCurrency(BitCoin) using ElasticNet (Linear regression with combined L1 and L2 priors as regularizers).
The app predicts the Closing price for BitCoin; the deployment is with real-time inference selecting the model trained on the "Close" column. However, this is a sample project and not to be tried for commercial purpose.

I have used pickling to serilize the model into byte code and store that; while deploying de-serilization is needed.

However, sklearn-onnx converts models in ONNX format which can be then used to compute predictions with the backend of our choice and this is the preferred deployment strategy. ONNX runtime support multiple languages and simplifies interoperanility.  There exists a way to automatically check every converter with onnxruntime, onnxruntime-gpu. Every converter is tested with this backend.

The notebook version here covers the details on ONNX deployment strategy.

More details can be found http://onnx.ai/sklearn-onnx/

# ML Workflow

Data Ingestion -> Data Validation -> Feature Engineering -> Time-series Split -> Model Training/Testing -> Accuracy Metrics -> Model Registration & Versioning -> Model Serving
