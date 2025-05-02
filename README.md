# Reconstruction-Based Fault Classification

<picture style="align: center; padding-bottom: 3mm;">
  <img alt="" src="./data/fault_isolation_abstract_procedure.PNG">
</picture>

This repository contains code required to participate in the [LiU-ICE DeluXe Diagnosis Competition](https://vehsys.gitlab-pages.liu.se/dx25benchmarks/liuice/liuice_index).

**TLDR: use an ensemble of autoencoders to detect and isolate a fault in a cyber-physical system**

Proposed method is **system-agnostic**, that it is doing not make any assumptions about the underlying system's dynamics, as long as there is a system simulation log per fault.

### High-level overview

### Training
1. Scale the data using a scaler of your choice (we use Standard scaler)
2. For each fault type train an autoencoder. You can vary the hyperparameters for each autoencoder, as long as they all use same scaled data. 
3. Compute reconstruction losses with trained autoencoder, and 98 percentile of reconstruction losses serves as an anomaly treshold (last step)

An advantage of this approach when compared to classification with a NN is that you might choose to train a custom autoencoder per fault type.
This can be seen in our training setup, in which we varied the size and training process of the autoencoder based on the task:
For more details about each model, check [trained_models](trained_models).

### Diagnosis
1. For each new sample, compute reconstruction losses with all autoencoders
2. Fault is isolated with the smallest loss
3. However, fault might also be of an unknown type. If the smallest reconstruction loss is above anomaly threshold for its autoencoder (98 percentile), treat the fault as "Unknown fault"

## Structure and Results

Install:
```commandline
pip install -r requirements.txt
```

[train_aes.py](train_aes.py) is used for training of autoencoders for all faults. It also computes mean, stddev, and 98 percentiles of reconstruction losses and saves it into [trained_models](trained_models).

[ae_based_diagnosis.py](ae_based_diagnosis.py) implements the competition interface.

[evaluation.py](evaluation.py) evaluates the approach, and outputs following confusion matrix:

<picture style="align: center; padding-bottom: 3mm;">
  <img alt="" src="./data/confusion_matrix.PNG">
</picture>

Category *Other* shows the classification rate of faults to unknown faults. **Note that for other we do not have actual samples**.

**Time statistics:** Average time 0.0011 Max time 0.0039 (Intel(R) Core(TM) Ultra 5 125U (14 CPUs), ~1.3GHz)
