import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss

from autoencoder import load_autoencoder


# detection (dim 1, boolean) - is a flag stating when a fault has been detected (detection = 0 if no fault and 1 if fault).
# isolation (dim 1x5, float) - non-negative ranking of diagnosis candidates in the following order:
# (f_pic,f_pim,f_waf,f_iml,f_x) where f_x is the ranking that the detected fault is none of the known faults.
# If detection = 0, the sum of elements in isolation = 0 and if detection = 1, the sum of elements in isolation = 1.


class AutoencoderBasedDiagnosis:
    def __init__(self):
        self.scaler = None
        self.aes = {}

        self.out_of_distribution_hits = 0

    def Initialize(self):

        save_folder = 'trained_models/'

        self.scaler = joblib.load('trained_models/scaler.sk')

        assert self.scaler is not None
        print("Scaler successfully loaded")

        for ae_name in ['f_iml', 'f_pic', 'f_pim', 'f_waf', 'NF']:
            self.aes[ae_name] = load_autoencoder(save_folder + ae_name, epoch=None)
            self.aes[ae_name].eval()

    def Input(self, sample):
        sample.drop(columns=['time'], inplace=True)

        scaled_sample = torch.tensor(self.scaler.transform(sample), dtype=torch.float32)

        # save losses of all AEs
        losses = []
        # save weather the reconstruction loss is beyond nominal threshold for each AE
        above_anomaly_threshold = []

        for ae_name, ae in self.aes.items():
            reconstruction = ae(scaled_sample)
            loss = mse_loss(reconstruction, scaled_sample).item()
            losses.append(- loss)
            above_anomaly_threshold.append(loss >= ae.reconstruction_loss_metrics['percentile_98'])

        min_loss_idx = losses.index(max(losses))

        if above_anomaly_threshold[min_loss_idx]:
            self.out_of_distribution_hits += 1

        if min_loss_idx == 4:
            nominal_behaviour = True
            isolation = np.zeros((1, 5))
        else:
            nominal_behaviour = False

            # if reconstruction loss of the detected fault is within reconstruction loss
            x = torch.tensor(losses[:4])
            # Apply softmax with scaling
            temperature = 0.2
            isolation = F.softmax(x / temperature, dim=0).numpy()
            isolation = np.append(isolation, 0)
            assert np.argmax(isolation) == min_loss_idx

            if above_anomaly_threshold[min_loss_idx]:
                max_val = isolation[min_loss_idx]
                new_fault = [0] * 5
                new_fault[min_loss_idx] = 1 - max_val
                new_fault[4] = max_val
                isolation = np.array(new_fault)

            assert 0.99 <= sum(isolation) <= 1.01

        # conform to competition format
        nominal_behaviour = np.array([nominal_behaviour], dtype=bool)
        isolation = isolation.reshape((1,5))

        return nominal_behaviour, isolation

