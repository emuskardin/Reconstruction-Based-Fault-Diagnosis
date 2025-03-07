# Define the VAE architecture

import torch
from torch import nn


def to_hidden_layers(hidden_dimension):
    decoder_layers = []
    for i in range(len(hidden_dimension) - 1):
        decoder_layers.append(nn.Linear(hidden_dimension[i], hidden_dimension[i + 1]))
        decoder_layers.append(nn.ReLU())
    # remove the last RELU
    decoder_layers.pop()
    return decoder_layers


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dimension, device=None):
        # last element of hidden dimensions is a latent dim
        super(AutoEncoder, self).__init__()
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'basic'
        self.metadata = {'type': self.name, 'input_dim': input_dim, 'hidden_dimension': hidden_dimension.copy()}

        hidden_dimension.insert(0, input_dim)

        # encoder
        encoder_layers = to_hidden_layers(hidden_dimension)
        self.encoder = nn.Sequential(*encoder_layers)

        # reverse layers for decoder
        hidden_dimension.reverse()

        # decoder
        decoder_layers = to_hidden_layers(hidden_dimension)
        self.decoder = nn.Sequential(*decoder_layers)

        # to gpu/cpu
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        # Decode
        dec_output = self.decoder(z)

        return dec_output
