import json
from statistics import mean, stdev

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from autoencoder import AutoEncoder, train_autoencoder, get_data_mean_squared_errors


class DxDataset(Dataset):

    def __init__(self, input_data, device=None):
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = torch.tensor(input_data.to_numpy(), dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

def get_all_data(path, fields_to_keep=None):
    file_path = path
    df = pd.read_csv(file_path)
    if fields_to_keep:
        df = df[fields_to_keep]
    return df

training_folder = 'data/training_data/'
save_path = 'trained_models/'

data_per_fault = {'f_iml' : ['wltp_f_iml_6mm.csv'],
                  'f_pic' : ['wltp_f_pic_090.csv', 'wltp_f_pic_110.csv'],
                  'f_pim' : ['wltp_f_pim_080.csv', 'wltp_f_pim_090.csv'],
                  'f_waf' : ['wltp_f_waf_105.csv', 'wltp_f_waf_110.csv'],
                  'NF' : ['wltp_NF.csv']}

# First, load all data from all fault types to fit a single global scaler
all_fault_data = []
for fault_type, files in data_per_fault.items():
    for f in files:
        df = get_all_data(training_folder + f)

        if fault_type != 'NF':
            df.query("time >= 115", inplace=True)

        df.drop(columns=['time'], inplace=True)
        all_fault_data.append(df)

# Create a single scaler instance for ALL fault types
scaler = RobustScaler()

# Fit the scaler on all data across all fault types
combined_data = pd.concat(all_fault_data, ignore_index=True)
scaler.fit(combined_data)

joblib.dump(scaler, 'trained_models/scaler.sk')

for fault_type, files in  data_per_fault.items():
    print('----------------------------------------')
    print(fault_type)


    # Load all files, filter, and concatenate
    all_data = pd.concat([get_all_data(training_folder + f) for f in files], ignore_index=True)

    if fault_type != 'NF':
        all_data.query("time >= 118", inplace=True)

    train_df, test_df = train_test_split(all_data, test_size=0.05, shuffle=True)
    # train_df = all_data

    # test_df.to_csv(f'data/test_data/test_set_{fault_type}.csv', index=False)

    train_df.drop(columns=['time'], inplace=True)
    scaled_data = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)

    data_set = DxDataset(scaled_data)
    num_features = train_df.shape[1]
    ae = AutoEncoder(input_dim=num_features, hidden_dimension=[64, 32, 16,])

    # train autoencoder
    train_autoencoder(ae, DataLoader(data_set, batch_size=16, shuffle=True), 15,
                      model_name=fault_type, save_path=save_path, save_every=None)

    # extract nominal loss and save to metadata json
    nominal_losses = get_data_mean_squared_errors(ae, data_set)

    with open(f'{save_path}/{fault_type}/metadata.json', "w", encoding="utf-8") as f:
        ae.metadata['mean_loss'] = mean(nominal_losses)
        ae.metadata['standard_dev'] = stdev(nominal_losses)
        ae.metadata['percentile_90'] = np.percentile(nominal_losses, 90)
        ae.metadata['percentile_95'] = np.percentile(nominal_losses, 95)
        ae.metadata['percentile_98'] = np.percentile(nominal_losses, 98)

        ae.metadata['max_loss'] = max(nominal_losses)

        json.dump(ae.metadata, f, indent=4)

