# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:32:04 2023

@author: binata
"""

"""
This is the script for 4-hr seq2seq LSTM Forecasting Model:
This network uses ELV, TWI, DTW + last 4 (RAINFALL, TIDE and WATERDEPTH) timesteps + future 4 (RAINFALL, and TIDE) timesteps to predict future 4 WATERDEPTH timesteps for flood-prone streets of Norfolk, VA

    flood-prone streets = 22
    n_back=4 and n_ahead = 4
    input = ELV, TWI, DTW,
            RH (past [t-3, t-2, t-1, t] and future [t+1, t+2, t+3, t+4] timesteps),
            TD (past [t-3, t-2, t-1, t] and future [t+1, t+2, t+3, t+4] timesteps), and
            w_depth (past [t-3, t-2, t-1, t] timesteps) 
    output = w_depth (future [t+1, t+2, t+3, t+4] timesteps)

It loads node_data, tide_data and weather_data from the relational database and prepares 3D tensor train and test data using lstm_data_tools.py for 22 flood-prone streets.
Then it hypertunes the model from a set of hyperparameters using the Bayesian optimization technique and then saves the best model and hyperparameters.

For 8-hr forecasting model, replace => n_ahead = 4 with n_ahead = 8
For 4-hr forecasting model w/o wl features, replace => x_cols = ['w_depth','ELV', 'DTW', 'TWI'] with x_cols = ['ELV', 'DTW', 'TWI']
For 4-hr forecasting model w/o spatial features, replace => x_cols = ['w_depth','ELV', 'DTW', 'TWI'] with x_cols = ['w_depth']

"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import kerastuner
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import load_model

import time


# set random seeds
np.random.seed(1)
tf.random.set_seed(1)


os.getcwd()
os.chdir('../../Street_Level_Flooding_1/')

import lstm_data_tools as ldt

db = ldt.SLF_Data_Builder(os.getcwd() + '/relational_data/')

#define model run_name
trial_model='seq2seqLSTM'
trial_data ='D0_R40_S22'
trial_param ='P6'
trial_time= '44'
trial_loss ='MSE'
trial_hp ='Bayes'


trial_all = '{}_{}_{}_{}_{}_{}'.format(trial_model, trial_data, trial_param, trial_time, trial_loss, trial_hp)
print(trial_all)

import mlflow
#mlflow experiment
experiment_name = "SLF_June"
experiment = mlflow.get_experiment_by_name(f"{experiment_name}")
mlflow.set_experiment(f"{experiment_name}")
mlflow.set_tracking_uri("file://C:/Users/hydrology.DESKTOP-S8EA36P/AppData/Roaming/Python/Python39/Scripts/mlruns")

with mlflow.start_run(run_name=trial_all):

    '///////////////////////////////////////////.........DATA.........///////////////////////////////////////////'
    
    'train_df and test_df'
    #specify parameters
    cols = ['FID_', 'Event', 'DateTime', 'RH', 'TD_HR', 'MAX15', 'HR_72', 'HR_2', 'w_depth', 'ELV', 'DTW', 'TWI']           
    print("Data Columns: ", cols)
    
    #specify events          
    Events =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24]   
    train_Events=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] 
    test_Events=[21, 22, 23, 24]
    
    #specify FIDs
    path_FIDs="input/"
    FID_selected=pd.read_csv(path_FIDs+"D0_R40_S22.csv") #flood-prone streets
    FIDs=FID_selected['FID_']
    
    train_nodes = FIDs
    train_events = train_Events
      
    test_nodes = FIDs
    test_events = test_Events
    
    mlflow.log_param("num_training_events", len(train_events))
    mlflow.log_param("num_training_nodes", len(train_nodes))
    mlflow.log_param("num_test_events", len(test_events))
    mlflow.log_param("num_test_nodes", len(test_nodes))

    #full data
    data_org = db.get_data(nodes = FIDs, events = Events, columns=cols)
    
    train_data_org = db.get_data(nodes = train_nodes, events = train_events, columns=cols)
    train_data_org.head()
    test_data_org = db.get_data(nodes = test_nodes, events = test_events, columns=cols)
    test_data_org.head()
    
    cols2scale = ['RH', 'TD_HR', 'w_depth', 'ELV', 'DTW', 'TWI']
    mlflow.log_param("num_params", len(cols2scale))
    mlflow.log_param("name_params", cols2scale)
    db.fit_scaler(train_data_org, columns_to_fit=cols2scale, scaler_type='Standard')
    train_data = db.scale_data(train_data_org, columns_to_scale=cols2scale)
    train_data.head()
    test_data = db.scale_data(test_data_org, columns_to_scale=cols2scale)
    test_data.head()
    
    print(len(train_events), len(test_events))
    
    lstm_train_data = ldt.SLF_LSTM_Data(train_data)
    lstm_test_data = ldt.SLF_LSTM_Data(test_data)
    
    n_back = 4
    n_ahead = 4
    mlflow.log_param("n_back", n_back)
    mlflow.log_param("n_ahead", n_ahead)
    forecast_cols = ['RH', 'TD_HR']
    x_cols = ['w_depth','ELV', 'DTW', 'TWI']
    y_cols = ['w_depth']
    
    lstm_train_data.build_data(
    n_back = n_back,
    n_ahead = n_ahead,
    forecast_cols = forecast_cols,
    y_cols = y_cols,
    x_cols = x_cols,
    verbose = False
    )
    
    lstm_test_data.build_data(
    n_back = n_back,
    n_ahead = n_ahead,
    forecast_cols = forecast_cols,
    y_cols = y_cols,
    x_cols = x_cols,
    verbose = False
    )
    
    train_x, train_y = lstm_train_data.get_lstm_data()
    test_x, test_y = lstm_test_data.get_lstm_data()
    
    #float32
    train_x = np.asarray(train_x).astype(np.float32)
    train_y = np.asarray(train_y).astype(np.float32)
    
    test_x = np.asarray(test_x).astype(np.float32)
    test_y = np.asarray(test_y).astype(np.float32)
    
    print('Data Shapes')
    print('Train x:', train_x.shape)
    print('Train y:', train_y.shape)
    print('Test x:', test_x.shape)
    print('Test y:', test_y.shape)
    
    
    #reshape target Y from 2D to 3D
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    

    '///////////////////////////////////////////.........MODEL.........///////////////////////////////////////////'
    
    def create_model(hp):
    
        # configure LSTM network hyperparameters
        n_back = 4
        n_ahead = 4
        num_units1 = 200
        act1 = 'relu'
        optimizer='Adam'
        lr=1e-2
        dp_rate=0.1
            
        num_units1 = hp.Choice('num_units1',values=[24, 50, 64, 75, 128, 200, 256, 512])
        act1 = hp.Choice('act1', values=['relu', 'tanh', 'sigmoid', 'selu'])
        lr = hp.Choice('learning_rate', values=[0.0005, 0.001, 0.005, 0.01])
        optimizer = hp.Choice('optimizer', values=['SGD', 'RMSprop', 'Adam', 'Nadam'])
        dp_rate =hp.Choice('dropout_rate', values=[0.05, 0.1, 0.15, 0.2])
 
        if optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        elif optimizer == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        else:
            raise
        
        # create the seq2seq LSTM model
        model = Sequential()
        
        #encoder
        model.add(LSTM(units=num_units1, activation=act1, input_shape=(None, train_x.shape[2]), use_bias=True))
        model.add(Dropout(rate=dp_rate))
        model.add(RepeatVector(n_ahead))
        
        #decoder
        model.add(LSTM(units=num_units1, activation=act1, return_sequences=True))
        model.add(Dropout(rate=dp_rate))
        model.add(tf.keras.layers.TimeDistributed(Dense(1)))
        
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'] ) 

        return model
    
    
    'custom Tuner'
    # model with hp tuning
    class CustomTuner(kerastuner.BayesianOptimization):
      def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[32,64])
        return super(CustomTuner, self).run_trial(trial, *args, **kwargs)
     
    #customer tuner
    tuner = CustomTuner(create_model, 
                    objective='val_loss',
                    max_trials=75,
                    directory='1.Run', 
                    project_name=trial_all,
                    overwrite=True)
    

    print(tuner.search_space_summary())
    
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    
    #kerastuner
    tuner.search(
    train_x, train_y,
    validation_data=(test_x, test_y),
    epochs=200, callbacks=[es], verbose=True) 
    
    
    mlflow.log_param("Bayes_max_trials", 75)
    mlflow.log_param("EarlyStopping", 15)
    mlflow.log_param("Epoch", 200)
    

    print('best trials')
    print(tuner.results_summary(1)) #best trial                
    
    #best1 model
    model = tuner.get_best_models(num_models=1)[0]
    model.summary()
    
    #save model
    model.save(f"1.Result/best1 {trial_model} {trial_data} {trial_param} {trial_time}_{trial_loss}_{trial_hp}.h5")
    print("Saved hypertuned best1 to disk")
    
    
    # log the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)
    print("best_hps:", best_hps)
    
    # log the best1 hyperparameters
    print("best1_hps:", best_hps[0])
    print("best1_hps:", best_hps[0].values)
    
    best1_hps = eval(str(best_hps[0].values))
    best1_num_units = best1_hps['num_units1']
    best1_dropout = best1_hps['dropout_rate']
    best_batch_size = best1_hps['batch_size']
    best1_hps = {k+"1": v for k, v in best1_hps.items()}
    mlflow.log_params(best1_hps)
    best1_hps_df = pd.DataFrame.from_dict(best1_hps, orient="index")
    best1_hps_df.to_csv(f"1.Result/best1_hps_df {trial_model} {trial_data} {trial_param} {trial_time}_{trial_loss}_{trial_hp}.csv")

    # serialize weights to HDF5
    model.save_weights(f"1.Result/best1 weights {trial_model} {trial_data} {trial_param} {trial_time}_{trial_loss}_{trial_hp}.h5")
    print("Saved best1 weights to disk")
    

  

    mlflow.end_run()    
        
