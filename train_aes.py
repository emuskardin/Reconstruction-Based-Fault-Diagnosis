import os
import json
from statistics import mean, stdev

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
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

faults = ['f_iml', 'f_pic', 'f_pim', 'f_waf', 'NF']
data_per_fault = {key: [] for key in faults}

# Path to the folder
folder_path = 'data/training_data'

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    for key in faults:
        if key in filename:
            data_per_fault[key].append(filename)

# First, load all data from all fault types to fit a single global scaler
all_fault_data = []
for fault_type, files in data_per_fault.items():
    for f in files:
        df = get_all_data(training_folder + f)
        df.drop(columns=['time'], inplace=True)
        all_fault_data.append(df)

# Create a single scaler instance for ALL fault types
scaler = StandardScaler()

# Fit the scaler on all data across all fault types
combined_data = pd.concat(all_fault_data, ignore_index=True)
scaler.fit(combined_data)

joblib.dump(scaler, 'trained_models/scaler.sk')

for fault_type, files in  data_per_fault.items():
    if fault_type not in {'f_pim', 'f_waf'}:
        continue
    print('----------------------------------------')
    print(fault_type)

    # Load all files, filter, and concatenate
    all_data = pd.concat([get_all_data(training_folder + f) for f in files], ignore_index=True)

    if fault_type != 'NF':
        all_data.query("time >= 118", inplace=True)

    # train_df, test_df = train_test_split(all_data, test_size=0.05, shuffle=True)
    train_df = all_data

    # test_df.to_csv(f'data/test_data/test_set_{fault_type}.csv', index=False)

    train_df.drop(columns=['time'], inplace=True)
    scaled_data = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)

    data_set = DxDataset(scaled_data)
    num_features = train_df.shape[1]

    ae_size = [16, 8, 4,]
    if fault_type in {'f_pim', 'f_waf'}:
        ae_size = [64, 32, 16, 8]
    ae = AutoEncoder(input_dim=num_features, hidden_dimension=ae_size)

    # train autoencoder
    train_autoencoder(ae, DataLoader(data_set, batch_size=128, shuffle=True), 30,
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

