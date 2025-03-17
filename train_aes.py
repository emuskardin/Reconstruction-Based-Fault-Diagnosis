import json
import pickle
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from autoencoder import AutoEncoder, train_autoencoder, get_data_mean_squared_errors
from utils import plot_reconstruction_loss


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

training_folder = 'data/trainingdata/'
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
        df = get_all_data(training_folder + f).query("time >= 115")
        df.drop(columns=['time'], inplace=True)
        all_fault_data.append(df)

# Create a single scaler instance for ALL fault types
scaler = StandardScaler()

# Fit the scaler on all data across all fault types
combined_data = pd.concat(all_fault_data, ignore_index=True)
scaler.fit(combined_data)

with open('trained_models/standardScaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

for fault_type, files in  data_per_fault.items():
    print('----------------------------------------')
    print(fault_type)

    # Load all files, filter, and concatenate
    all_data = [get_all_data(training_folder + f).query("time >= 115") for f in files]

    for d in all_data:
        d.drop(columns=['time'], inplace=True)

    window_size = 1
    num_features = all_data[0].shape[1]

    # Scale each DataFrame individually while preserving the original structure
    scaled_data = pd.concat([pd.DataFrame(scaler.transform(df), columns=df.columns) for df in all_data],
                            ignore_index=True)

    data_set = DxDataset(scaled_data)

    ae = AutoEncoder(input_dim=num_features, hidden_dimension=[64, 32, 16])

    # train autoencoder
    train_autoencoder(ae, DataLoader(data_set, batch_size=16, shuffle=True), 10,
                      model_name=fault_type, save_path=save_path)

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

    plot_reconstruction_loss(save_path + fault_type, nominal_losses, ae.metadata['mean_loss'],
                             ae.metadata['standard_dev'], ae.metadata['percentile_95'])

    # Normalizer does not work
    # 32 batch, StandardScaler, and 32, or 16 initial worked well
    # MinMax scaler + 64 model -> works well!
    # Standard scaler, shuffle 32 batch, + 64 model -> works very well!

