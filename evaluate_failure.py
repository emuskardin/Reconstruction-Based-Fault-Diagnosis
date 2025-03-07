import os
import pickle
import random
import time
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss

from autoencoder_common import load_autoencoder

# detection (dim 1, boolean) - is a flag stating when a fault has been detected (detection = 0 if no fault and 1 if fault).
# isolation (dim 1x5, float) - non-negative ranking of diagnosis candidates in the following order:
# (f_pic,f_pim,f_waf,f_iml,f_x) where f_x is the ranking that the detected fault is none of the known faults.
# If detection = 0, the sum of elements in isolation = 0 and if detection = 1, the sum of elements in isolation = 1.


def load_csv_files_from_folder(folder_path):
    dataframes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            dataframes[file_name] = df

    return dataframes

def power_normalization(x, alpha=0.5):
    x_pow = np.power(x, alpha)  # Raise to the power
    return x_pow / np.sum(x_pow)  # Normalize

scaler = None
aes = {}

save_folder = 'trained_models/'

with open(save_folder + 'standardScaler.pickle', 'rb') as handle:
    scaler = pickle.load(handle)

assert scaler is not None
print("Scaler successfully loaded")

for ae_name in ['f_iml', 'f_pic', 'f_pim', 'f_waf', 'NF']:
    aes[ae_name] = load_autoencoder(save_folder + ae_name)

def make_prediction(scaler, aes, sample):
    sample.drop(columns=['time'], inplace=True)

    scaled_sample = torch.tensor(scaler.transform(sample), dtype=torch.float32)

    losses = []
    for ae_name, ae in aes.items():
        reconstruction = ae(scaled_sample)
        loss = - mse_loss(reconstruction, scaled_sample).item()
        losses.append(loss)

    min_loss_idx = losses.index(max(losses))

    if min_loss_idx == 4:
        nominal_behaviour = True
        isolation = np.zeros((1,5))
    else:
        nominal_behaviour = False

        # Example tensor
        x = torch.tensor(losses[:4])

        # Apply softmax
        softmax_result = F.softmax(x, dim=0).numpy()
        isolation = softmax_result

    # TODO remove this in future
    # Example tensor
    x = torch.tensor(losses[:4])

    # Apply softmax
    softmax_result = F.softmax(x, dim=0).numpy()
    print(softmax_result)

    isolation = softmax_result

    return nominal_behaviour, isolation


test_data = load_csv_files_from_folder('data/trainingdata')

# Confusion matrix structure
confusion_matrix = np.zeros((5, 5), dtype=int)  # Assuming 5 categories

# Dataset mapping (order is important for indexing)
dataset_mapping = ['f_iml', 'f_pic', 'f_pim', 'f_waf', 'NF']

avg_time = []
for data_set_name, data in test_data.items():
    num_samples = data.shape[0]

    num_test_per_category = 10
    num_test = num_test_per_category if 'NF' in data_set_name or 'f_iml' in data_set_name else num_test_per_category//2
    for _ in range(num_test):  # Run multiple test iterations
        sample = data.iloc[random.randint(0, num_samples - 1)].to_frame().transpose()

        s = time.time()
        nominal_behaviour, isolation = make_prediction(scaler, aes, sample)
        avg_time.append(time.time() - s)
        # Find the actual dataset category
        actual_category = next((dataset_mapping.index(key) for key in dataset_mapping if key in data_set_name), None)

        if actual_category is not None:
            if actual_category == dataset_mapping.index('NF'):
                predicted_category = dataset_mapping.index('NF') if nominal_behaviour else np.argmax(isolation)  # -1 if misclassified
            else:
                predicted_category = np.argmax(isolation)  # Get the predicted index

            if 0 <= predicted_category < len(dataset_mapping):
                confusion_matrix[actual_category, predicted_category] += 1  # Update confusion matrix


df_cm = pd.DataFrame(confusion_matrix, index=dataset_mapping, columns=dataset_mapping)

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor="black")
plt.title("Confusion Matrix for Classification Results")
plt.ylabel("Actual Category")
plt.xlabel("Predicted Category")
plt.show()

print('Average prediction time', mean(avg_time), 'Max time', max(avg_time))