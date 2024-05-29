# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:32:04 2023

@author: binata
"""

"""
This is the script for 4-hr seq2seq LSTM Forecasting Model:
This network uses ELV, TWI, DTW + last 4 (Rainfall, Tide and Waterdepth) timesteps + future 4 (Rainfall, and Tide) timesteps to predict future 4 Waterdepth timesteps for flood-prone streets of Norfolk, VA

    flood-prone streets = 22
    n_back=4 and n_ahead = 4
    input = ELV, TWI, DTW,
            RH (past [t-3, t-2, t-1, t] and future [t+1, t+2, t+3, t+4] timesteps),
            TD (past [t-3, t-2, t-1, t] and future [t+1, t+2, t+3, t+4] timesteps), and
            w_depth (past [t-3, t-2, t-1, t] timesteps) 
    output = w_depth (future [t+1, t+2, t+3, t+4] timesteps)

It loads node_data, tide_data and weather_data from the relational database and prepares 3D tensor train and test data using lstm_data_tools.py for 22 flood-prone streets.
Then it loads the best model, predicts future water depth on train and test data and saves predictions to CSV files. It also plots water depth from seq2seq LSTM and ground-truth TUFLOW for 6 streets.

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
    


  
    '///////////////////////////////////////////.........PREDICTION.........///////////////////////////////////////////'
    
    hp_model='best1'
    
    # Load the model
    model = load_model('best1 seq2seqLSTM D0_R40_S22 P6 44_MSE_Bayes.h5')
    
    'train'
    preds = model.predict(train_x)

    #new
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[2] * train_y.shape[1]))
    preds = preds.reshape((preds.shape[0], preds.shape[2] * preds.shape[1]))
    
    
    rmse = np.sqrt(np.mean((preds - train_y)**2))
    mlflow.log_metric(f"{hp_model} train_rmse", rmse)
    print(f"Train RMSE for {hp_model} {trial_all}: {rmse}")
    
    
    train_data_1=train_data
    
    'remap multi-ahead'
    for k in range(n_ahead):
        preds_col = pd.Series(preds[:,k], index=lstm_train_data.data_map)
        train_data_1[f'preds_y{k+1}_s'] = preds_col
        train_data_1[f'preds_y{k+1}'] = train_data_1[f'preds_y{k+1}_s'].shift(k)
        del train_data_1[f'preds_y{k+1}_s']

        real_col = pd.Series(train_y[:,k], index=lstm_train_data.data_map)
        train_data_1[f'real_y{k+1}_s'] = real_col
        train_data_1[f'real_y{k+1}'] = train_data_1[f'real_y{k+1}_s'].shift(k)
        del train_data_1[f'real_y{k+1}_s']
    
    train_data_1_inv = train_data_1.copy()
    train_data_1_inv.head()

    cols2scale = ['RH','w_depth', 'preds_y1', 'real_y1', 'preds_y2',
    'real_y2', 'preds_y3', 'real_y3', 'preds_y4', 'real_y4']
    orig_cols = ['RH','w_depth', 'w_depth', 'w_depth', 'w_depth', 
    'w_depth', 'w_depth', 'w_depth', 'w_depth', 'w_depth']


    # inverse scale the train data
    train_data_1_inv = db.inverse_scale_data(train_data_1_inv, columns_to_scale=cols2scale, orig_col_names=orig_cols)
    train_data_1_inv.head()


    'plot for 6 FIDs'
    plot_std = False
    events_to_plot=train_events
    nodes_to_plot = [11503, 12035, 12205, 12591, 13128, 13348]
    
    for t in range(1, n_ahead+1):
        for event_id in events_to_plot:
            fig, axes = plt.subplots(
                nrows=1, 
                ncols=len(nodes_to_plot), 
                figsize=(len(nodes_to_plot)*3, 3), 
                sharey=True)
            for i, node_id in enumerate(nodes_to_plot):
                df_subset = train_data_1_inv[(train_data_1_inv.Event == event_id) & (train_data_1_inv.FID_ == node_id)]
                real_y = df_subset[f'w_depth']
                preds_y = df_subset[f'preds_y{t}']
                axes[i].set_title(f'FID:{node_id}')
                x_vals = np.arange(preds_y.shape[0])
                axes[i].plot(x_vals, real_y, label=f'Real{t}')
                axes[i].plot(x_vals, preds_y, label=f'LSTM{t}')
                axes[i].plot(x_vals, df_subset[f'real_y{t}'], ':k', label=f'InputY{t}')
                axes[i].legend()
    
            fig.suptitle(f't+{t} Event:{event_id}', y=1.05)  
            plt.savefig(f"1.Result/figs_ED_new/{hp_model}_train_t+{t}_Event_{event_id}.png")
            mlflow.log_artifact(f"1.Result/figs_ED_new/{hp_model}_train_t+{t}_Event_{event_id}.png")       
        
    'dataframe to csv'
    print ('dataframe to csv')   
 
    train_data_1_inv=train_data_1_inv.reset_index(drop=True)
    train_data_1_inv['Datetime'] =  pd.to_datetime(train_data_1_inv['DateTime'], format='%b%y_%d_%H')
    train_data_1_inv.set_index('Datetime', inplace=True, drop=True)
     
    #csv    
    train_data_1_inv.to_csv(f"1.Result/stat_ED_new/All_train_{trial_model}_{hp_model}_{trial_data}_{trial_param}_{trial_time}_{trial_loss}_{trial_hp}.csv")
    mlflow.log_artifact(f"1.Result/stat_ED_new/All_train_{trial_model}_{hp_model}_{trial_data}_{trial_param}_{trial_time}_{trial_loss}_{trial_hp}.csv")
        
    start = time.time()  
    
    'test'
    preds = model.predict(test_x)
    
    end = time.time()
    pred_time = (end - start)/60
    print("pred time: ", pred_time)
    mlflow.log_metric(f"{hp_model} pred_time", pred_time)
    
    #new
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[2] * test_y.shape[1]))
    preds = preds.reshape((preds.shape[0], preds.shape[2] * preds.shape[1]))


    rmse = np.sqrt(np.mean((preds - test_y)**2))
    mlflow.log_metric(f"{hp_model} test_rmse", rmse)
    print(f"Test RMSE for {hp_model} {trial_all}: {rmse}")

    test_data_1=test_data
        
    'remap multi-ahead'
    for k in range(n_ahead):
        preds_col = pd.Series(preds[:,k], index=lstm_test_data.data_map)
        test_data_1[f'preds_y{k+1}_s'] = preds_col
        test_data_1[f'preds_y{k+1}'] = test_data_1[f'preds_y{k+1}_s'].shift(k)
        del test_data_1[f'preds_y{k+1}_s']

        real_col = pd.Series(test_y[:,k], index=lstm_test_data.data_map)
        test_data_1[f'real_y{k+1}_s'] = real_col
        test_data_1[f'real_y{k+1}'] = test_data_1[f'real_y{k+1}_s'].shift(k)
        del test_data_1[f'real_y{k+1}_s']
    
    test_data_1_inv = test_data_1.copy()
    test_data_1_inv.head()
          
          
    cols2scale = ['RH','w_depth', 'preds_y1', 'real_y1', 'preds_y2',
    'real_y2', 'preds_y3', 'real_y3', 'preds_y4', 'real_y4']
    orig_cols = ['RH','w_depth', 'w_depth', 'w_depth', 'w_depth', 
    'w_depth', 'w_depth', 'w_depth', 'w_depth', 'w_depth']
    
    # inverse scale the test data
    test_data_1_inv = db.inverse_scale_data(test_data_1_inv, columns_to_scale=cols2scale, orig_col_names=orig_cols)
    test_data_1_inv.head()    
    
    
    'plot for 6 FIDs'
    plot_std = False
    events_to_plot=test_events
    nodes_to_plot = [11503, 12035, 12205, 12591, 13128, 13348]
    
    for t in range(1, n_ahead+1):
        for event_id in events_to_plot:
            fig, axes = plt.subplots(
                nrows=1, 
                ncols=len(nodes_to_plot), 
                figsize=(len(nodes_to_plot)*3, 3), 
                sharey=True)
            for i, node_id in enumerate(nodes_to_plot):
                df_subset = test_data_1_inv[(test_data_1_inv.Event == event_id) & (test_data_1_inv.FID_ == node_id)]
                real_y = df_subset[f'w_depth']
                preds_y = df_subset[f'preds_y{t}']
                axes[i].set_title(f'FID:{node_id}')
                x_vals = np.arange(preds_y.shape[0])
                axes[i].plot(x_vals, real_y, label=f'Real{t}')
                axes[i].plot(x_vals, preds_y, label=f'LSTM{t}')
                axes[i].plot(x_vals, df_subset[f'real_y{t}'], ':k', label=f'InputY{t}')
                axes[i].legend()
    
            fig.suptitle(f't+{t} Event:{event_id}', y=1.05)  
            plt.savefig(f"1.Result/figs_ED_new/{hp_model}_test_t+{t}_Event_{event_id}.png")
            mlflow.log_artifact(f"1.Result/figs_ED_new/{hp_model}_test_t+{t}_Event_{event_id}.png")
    
        
    'dataframe to csv'
    print ('dataframe to csv')   

    test_data_1_inv=test_data_1_inv.reset_index(drop=True)
    test_data_1_inv['Datetime'] =  pd.to_datetime(test_data_1_inv['DateTime'], format='%b%y_%d_%H')
    test_data_1_inv.set_index('Datetime', inplace=True, drop=True)
     
    #csv    
    test_data_1_inv.to_csv(f"1.Result/stat_ED_new/All_test_{trial_model}_{hp_model}_{trial_data}_{trial_param}_{trial_time}_{trial_loss}_{trial_hp}.csv") 
    mlflow.log_artifact(f"1.Result/stat_ED_new/All_test_{trial_model}_{hp_model}_{trial_data}_{trial_param}_{trial_time}_{trial_loss}_{trial_hp}.csv")       

    mlflow.end_run()    
        
