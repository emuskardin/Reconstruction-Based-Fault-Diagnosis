import pickle

import numpy as np
import torch
from torch.nn.functional import mse_loss
import torch.nn.functional as F


from autoencoder import load_autoencoder

# detection (dim 1, boolean) - is a flag stating when a fault has been detected (detection = 0 if no fault and 1 if fault).
# isolation (dim 1x5, float) - non-negative ranking of diagnosis candidates in the following order:
# (f_pic,f_pim,f_waf,f_iml,f_x) where f_x is the ranking that the detected fault is none of the known faults.
# If detection = 0, the sum of elements in isolation = 0 and if detection = 1, the sum of elements in isolation = 1.


class AutoencoderBasedDiagnosis:
    def __init__(self):
        self.scaler = None
        self.aes = {}

    def Initialize(self):

        save_folder = 'trained_models/'

        with open(save_folder + 'standardScaler.pickle', 'rb') as handle:
            self.scaler = pickle.load(handle)

        assert self.scaler is not None
        print("Scaler successfully loaded")

        for ae_name in ['f_iml', 'f_pic', 'f_pim', 'f_waf', 'NF']:
            self.aes[ae_name] = load_autoencoder(save_folder + ae_name)

    def Input(self, sample):
        sample.drop(columns=['time'], inplace=True)

        scaled_sample = torch.tensor(self.scaler.transform(sample), dtype=torch.float32)

        losses = []
        for ae_name, ae in self.aes.items():
            reconstruction = ae(scaled_sample)
            loss = - mse_loss(reconstruction, scaled_sample).item()
            losses.append(loss)

        min_loss_idx = losses.index(max(losses))

        if min_loss_idx == 4:
            nominal_behaviour = True
            isolation = np.zeros((1, 5))
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
        temperature = 0.1
        softmax_result = F.softmax(x / temperature, dim=0).numpy()
        if not nominal_behaviour:
            print(softmax_result)

        assert 0.99 <= sum(softmax_result) <= 1.01

        isolation = softmax_result

        return nominal_behaviour, isolation

