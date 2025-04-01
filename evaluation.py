import os
import random
import time
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ae_based_diagnosis import AutoencoderBasedDiagnosis

def load_csv_files_from_folder(folder_path):
    dataframes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            if 'NF' not in file_path:
                df.query("time >= 118", inplace=True)

            dataframes[file_name] = df

    return dataframes

data_set = 'test_data'
test_data = load_csv_files_from_folder(f'data/{data_set}')

# Confusion matrix structure
confusion_matrix = np.zeros((5, 6), dtype=int)  # Assuming 5 categories

# Dataset mapping (order is important for indexing)
dataset_mapping = ['NF', 'f_iml', 'f_pic', 'f_pim', 'f_waf', 'Other']

predictor = AutoencoderBasedDiagnosis()
predictor.Initialize()

avg_time = []
for data_set_name, data in test_data.items():
    num_samples = data.shape[0]
    # print(num_samples)

    num_test_per_category = 1000

    num_test = num_test_per_category if 'NF' in data_set_name or 'f_iml' in data_set_name or data_set == 'test_data' else num_test_per_category // 2

    # num_test = num_test_per_category

    for _ in range(num_test):
        sample = data.iloc[random.randint(0, num_samples - 1),:].to_frame().transpose()

        s = time.time()
        nominal_behaviour, isolation = predictor.Input(sample)
        avg_time.append(time.time() - s)

        # Find the actual dataset category
        actual_category = next((dataset_mapping.index(key) for key in dataset_mapping if key in data_set_name), None)
        if actual_category is None:
            continue

        if nominal_behaviour:
            predicted_category = dataset_mapping.index('NF')
        else:
            predicted_category = np.argmax(isolation) + 1 # +1 since NF is the first category in confusion matrix

        confusion_matrix[actual_category, predicted_category] += 1


print(f'Out of distribution hits: {predictor.out_of_distribution_hits}')

df_cm = pd.DataFrame(confusion_matrix, index=dataset_mapping[:-1], columns=dataset_mapping)

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor="black",)
plt.title("Confusion Matrix for Classification Results")
plt.ylabel("Actual Category")
plt.xlabel("Predicted Category")
plt.show()

print('Average prediction time', mean(avg_time), 'Max time', max(avg_time))