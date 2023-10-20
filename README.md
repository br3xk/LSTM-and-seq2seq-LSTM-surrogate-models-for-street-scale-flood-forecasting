# LSTM-and-seq2seq-LSTM-surrogate-models-for-street-scale-nuisance-flood-forecasting
This network uses ELV, TWI, DTW + last 4 RAINFALL, TIDE and WATERDEPTH + future 4 RAINFALL, and TIDE to predict future 4 WATERDEPTH for flood-prone streets of Norfolk, VA. 
The script loads node_data, tide_data and weather_data from the relational database and prepares 3D tensor train and test data using lstm_data_tools.py for 22 flood-prone streets. Then, it hypertunes the model from a set of hyperparameters using the Bayesian optimization technique and then saves the best model and hyperparameters. Finally, it predicts future water depth on train and test data and saves predictions to CSV files. It also plots water depth from LSTM and ground-truth TUFLOW for 6 streets.
The input data is available on Hydroshare (Roy, 2023).


References:

Roy, B. (2023). Input data for LSTM and seq2seq LSTM surrogate models for multi-step-ahead street-scale flood forecasting in Norfolk, VA, HydroShare, http://www.hydroshare.org/resource/e5d6d32a320f4bcca679e0bf388c2bcc
