# LSTM-and-seq2seq-LSTM-surrogate-models-for-street-scale-flood-forecasting
This network uses ELV, TWI, DTW + last 4 RAINFALL, TIDE and WATERDEPTH + future 4 RAINFALL, and TIDE to predict future 4 WATERDEPTH for flood-prone streets of Norfolk, VA.

It loads node_data, tide_data and weather_data from the relational database and prepares 3D tensor train and test data using lstm_data_tools.py for 22 flood-prone streets.

It hypertunes the model from a set of hyperparameters using the Bayesian optimization technique and then saves the best model and hyperparameters.

It predicts future water depth on train and test data and saves predictions to CSV files. It also plots water depth from LSTM and ground-truth TUFLOW for 6 streets.

N.B.

For 8-hr forecasting model, replace => n_ahead = 4 with n_ahead = 8

For 4-hr forecasting model w/o wl features, replace => x_cols = ['w_depth','ELV', 'DTW', 'TWI'] with x_cols = ['ELV', 'DTW', 'TWI']

For 4-hr forecasting model w/o spatial features, replace => x_cols = ['w_depth','ELV', 'DTW', 'TWI'] with x_cols = ['w_depth']

