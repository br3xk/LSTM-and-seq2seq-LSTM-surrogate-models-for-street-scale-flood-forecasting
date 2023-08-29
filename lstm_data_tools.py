# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:00:10 2023

@author: steven (original), binata (modified)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import re 
import os

class SLF_Data_Builder:
    def __init__(self, path_to_data):

        self.node_data = pd.read_csv(
            filepath_or_buffer = path_to_data + 'node_data.csv',
            index_col = 'FID_'
        )
        
        self.tide_data = pd.read_csv(
            filepath_or_buffer = path_to_data + 'tide_data.csv',
            index_col = 'DateTime'
        )

        self.weather_data = pd.read_csv(
            filepath_or_buffer = path_to_data + 'weather_data.csv',
            index_col = ['FID_', 'DateTime']
        )

        self.scaler_dict = {}

    def get_all_nodes(self):
        return self.weather_data.index.unique('FID_')

    def get_all_events(self):
        return self.weather_data['Event'].unique()

    def get_all_columns(self):
        node_cols = list(self.node_data.columns)
        tide_cols = list(self.tide_data.columns)
        weather_cols = list(self.weather_data.columns)
        index_cols = ['FID_', 'DateTime']
        return index_cols + node_cols + tide_cols + weather_cols

    def get_data(
        self,
        nodes = None,
        events = None,
        columns = None
    ):
        
        
        if nodes is None:
            df = self.weather_data
        else:
            df = self.weather_data.loc[nodes]

        if events is not None:
            df = df.loc[df.Event.isin(events)]

        # Join using the intersection of FID_s (avoids rows for missing FID_s)
        df = df.join(self.node_data, on = 'FID_', how = 'inner')

        # Use standard join (we want to see NaNs if we're missing data
        #   as missing timesteps could throw off data creation). 
        df = df.join(self.tide_data, on = 'DateTime')

        df.reset_index(inplace = True)

        if columns is not None:
            df = df[columns]

        return df

    # TODO: Allow for scaler kwargs to be passed to scalers
    def fit_scaler(self, df, columns_to_fit, scaler_type = 'standard'):
        for column in columns_to_fit:
            if scaler_type.lower() == 'standard':
                scaler = StandardScaler()
            elif scaler_type.lower() == 'robust':
                scaler = RobustScaler()
            else:
                raise Exception('Unsupported scaler_type')
            scaler.fit(df[column].to_numpy().reshape(-1,1))
            self.scaler_dict[column] = scaler

    def scale_data(self, df, columns_to_scale):
        df_copy = df.copy()
        for column in columns_to_scale:
            df_copy[column] = self.scaler_dict[column].transform(
                df[column].to_numpy().reshape(-1,1)
            )
        return df_copy

    def inverse_scale_data(self, df, columns_to_scale, orig_col_names = None):
        # If no names are given, assume the columns_to_scale match the original column names
        if orig_col_names is None:
            orig_col_names = columns_to_scale
        df_copy = df.copy()
        for orig_col, column in zip(orig_col_names, columns_to_scale):
            df_copy[column] = self.scaler_dict[orig_col].inverse_transform(
                df[column].to_numpy().reshape(-1,1)
            )
        return df_copy

from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

class SLF_LSTM_Data:
    def __init__(self, df) -> None:
        self.df = df
        self.lstm_dict = {}

    def __get_xy_column_indices(
        self,
        x_cols, 
        y_cols, 
        forecast_cols, 
        n_ahead,
        verbose
    ):

        column_names = list(self.df.columns)

        initial_size = len(column_names)

        for i in range(n_ahead):
            [column_names.append(x+'_' + str(i+1) + 'HR') for x in forecast_cols]

        x_indices = np.array([column_names.index(x) for x in x_cols + forecast_cols])
        
        # Add indices for forecast data
        x_indices = np.concatenate([x_indices, np.arange(initial_size, len(column_names))])

        y_indices = np.array([column_names.index(y) for y in y_cols])

        if verbose:
            print(f'Column Names: {column_names}')
            print(f'X_ind: {x_indices}, y_ind: {y_indices}')

        return x_indices, y_indices


    def __create_forecast_data(self, df, forecast_cols, n_ahead):
        n_fcols = len(forecast_cols)
        forecasted_data = np.zeros((df.shape[0], n_ahead*n_fcols))
        for i in range(n_ahead):
            new_data = df[forecast_cols].shift(-i-1, fill_value=np.nan).to_numpy()
            forecasted_data[:,i*n_fcols:(i+1)*n_fcols] = new_data

        return forecasted_data




    def get_lstm_data(self):
        x_data = []
        y_data = []
        data_map = []
        for key in self.lstm_dict:
            event, node = int(re.findall('Event:(\d+)', key)[0]), int(re.findall('FID:(\d+)', key)[0])
            x_data.append(self.lstm_dict[f'Event:{event}/FID:{node}'][0])
            y_data.append(self.lstm_dict[f'Event:{event}/FID:{node}'][1])
            data_map.append(self.lstm_dict[f'Event:{event}/FID:{node}'][2])

        self.x = np.concatenate(x_data, axis = 0)
        self.y = np.concatenate(y_data, axis = 0)
        self.data_map = np.concatenate(data_map, axis = 0)

        return self.x, self.y

    def build_data(
        self,
        n_back = 1,
        n_ahead = 1,
        forecast_cols = ['RH', 'TD_HR'],
        y_cols = ['w_depth'],
        x_cols = ['w_depth', 'ELV', 'DTW', 'TWI'],
        verbose = False
    ):

        x_ind, y_ind = self.__get_xy_column_indices(
            x_cols = x_cols,
            y_cols = y_cols,
            forecast_cols = forecast_cols,
            n_ahead = n_ahead,
            verbose = verbose
        )

        if len(y_ind) == 1 and verbose:
            print('Y array will be truncated to 2D')

        event_ids = self.df['Event'].unique()
        node_FIDs = self.df['FID_'].unique()

        for event in tqdm(event_ids):
            if verbose:
                print(f'Processing Event #{event}')
            event_data = self.df.loc[self.df.Event == event]
            for node in node_FIDs:
                event_node_string = f'Event:{event}/FID:{node}'
                node_data = event_data.loc[event_data.FID_ == node]

                forecasted_data = self.__create_forecast_data(node_data, forecast_cols, n_ahead)

                full_data = np.concatenate([node_data.to_numpy(), forecasted_data], axis = 1)
                
                window_shape = (n_back+n_ahead, full_data.shape[1])

                if verbose and full_data.shape[0] < window_shape[0]:
                    print(f'Skipping Event #{event} due to low samples')
                    break
                    
                if full_data.shape[0] > window_shape[0]:
                    lstm_data = sliding_window_view(full_data, window_shape=window_shape).squeeze()

                    #lstm_data now has only node_data.shape - n_back - n_ahead + 1 rows
                    x = lstm_data[:,:n_back,x_ind]
                    y = lstm_data[:,n_back:,y_ind]
                    if n_ahead == 1:
                        df_index = node_data.index[n_back:]
                    else:
                        df_index = node_data.index[n_back:-(n_ahead-1)]


                    if len(y_ind) == 1:
                        y = np.squeeze(y,axis=2)

                    self.lstm_dict[event_node_string] = (x,y,df_index)

def get_toy_nodes(db):
    lat_lon_start = (36.86168461124184, -76.30263385405442)
    deg_away = 0.005
    nodes = db.node_data[(db.node_data.LAT < lat_lon_start[0] + deg_away) & ((db.node_data.LAT > lat_lon_start[0] - deg_away))]
    nodes = nodes[(nodes.LON < lat_lon_start[1] + deg_away) & (nodes.LON > lat_lon_start[1] - deg_away)]
    return list(nodes.index)

def split_nodes_and_events(data, nodes, events, test_size = 0.1, random_state = 1):
    # Split nodes and events into train and test data
    train_nodes, _ = train_test_split(nodes, test_size = test_size, random_state = random_state)
    train_events, _ = train_test_split(events, test_size = test_size, random_state = random_state)
    train_data = data[data.FID_.isin(train_nodes)]
    train_data = train_data[train_data.Event.isin(train_events)]

    test_data = data[~data.index.isin(train_data.index)]

    return train_data, test_data

def get_toy_data(path = None):
    if path == None:
        path = os.getcwd()

    db = SLF_Data_Builder(path + '/relational_data/')

    # Get node data for a portion of Norfolk around the hospital
    allnodes = get_toy_nodes(db)
    allevents = db.get_all_events()

    cols = ['FID_', 'Event', 'RH', 'TD_HR', 'MAX15', 'HR_72', 'HR_2', 'w_depth', 'ELV', 'DTW', 'TWI']

    data = db.get_data(nodes = allnodes, events = allevents, columns=cols)
    
    train_data, test_data = split_nodes_and_events(data, allnodes, allevents)

    train_lstm_data_obj = SLF_LSTM_Data(train_data)
    test_lstm_data_obj = SLF_LSTM_Data(test_data)

    n_back = 4
    n_ahead = 1
    forecast_cols = ['RH', 'TD_HR']
    x_cols = ['w_depth', 'ELV', 'DTW', 'TWI', 'MAX15', 'HR_72', 'HR_2']
    y_cols = ['w_depth']

    train_lstm_data_obj.build_data(
        n_back = n_back,
        n_ahead = n_ahead,
        forecast_cols = forecast_cols,
        y_cols = y_cols,
        x_cols = x_cols,
        verbose = False
    )

    test_lstm_data_obj.build_data(
        n_back = n_back,
        n_ahead = n_ahead,
        forecast_cols = forecast_cols,
        y_cols = y_cols,
        x_cols = x_cols,
        verbose = False
    )

    return train_lstm_data_obj, test_lstm_data_obj