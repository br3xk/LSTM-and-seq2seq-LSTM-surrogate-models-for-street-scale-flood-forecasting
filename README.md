# LSTM-and-seq2seq-LSTM-surrogate-models-for-street-scale-flood-forecasting
4-hr LSTM and seq2seq LSTM Forecasting Model:
This network uses ELV, TWI, DTW + last 4 RAINFALL, TIDE and WATERDEPTH + future 4 RAINFALL, and TIDE to predict future 4 WATERDEPTH for flood-prone streets of Norfolk, VA

    flood-prone streets = 22
    n_back=4 and n_ahead = 4
    input = ELV, TWI, DTW,
            RH (past [t-3, t-2, t-1, t] and future [t+1, t+2, t+3, t+4] timesteps),
            TD (past [t-3, t-2, t-1, t] and future [t+1, t+2, t+3, t+4] timesteps), and
            w_depth (past [t-3, t-2, t-1, t] timesteps) 
    output = w_depth (future [t+1, t+2, t+3, t+4] timesteps)

It loads node_data, tide_data and weather_data from the relational database and prepares 3D tensor train and test data using lstm_data_tools.py for 22 flood-prone streets.
It hypertunes the model from a set of hyperparameters using the Bayesian optimization technique and then saves the best model and hyperparameters.
It predicts future water depth on train and test data and saves predictions to CSV files. It also plots water depth from LSTM and ground-truth TUFLOW for 6 streets.
