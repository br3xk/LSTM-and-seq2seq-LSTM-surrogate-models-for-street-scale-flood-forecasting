# LSTM-and-seq2seq-LSTM-surrogate-models-for-street-scale-flood-forecasting

This network uses topographic features (ELV, TWI, DTW) + last 4 (Rainfall, Tide and Waterdepth) timesteps + future 4 (Rainfall, and Tide) timesteps to predict future 4 Waterdepth timesteps for flood-prone streets of Norfolk, VA

1. "lstm_44_hyper.py" or "s2s_lstm_44_hyper.py" loads node_data, tide_data and weather_data from the relational database and prepares 3D tensor train and test data using "lstm_data_tools.py" for 22 flood-prone streets. Then, it hypertunes the model from a set of hyperparameters using the Bayesian optimization and then saves the best model and hyperparameters. 

2. "lstm_44_pred.py" or "s2s_lstm_44_pred.py" loads the best model, predicts future water depth for train and test data and saves predictions to CSV files. It also plots water depth from LSTM/seq2seq LSTM and ground-truth TUFLOW for 6 streets. 

The input data is available on Hydroshare (Roy et al., 2023).

References:

Roy, B. (2023). Input data for LSTM and seq2seq LSTM surrogate models for multi-step-ahead street-scale flood forecasting in Norfolk, VA, HydroShare, http://www.hydroshare.org/resource/e5d6d32a320f4bcca679e0bf388c2bcc
