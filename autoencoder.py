import json
import os

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader


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

        # used during prediction phase, populated when loading a model
        self.reconstruction_loss_metrics = {}

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


def load_autoencoder(path_to_folder):
    metadata = json.load(open(f'{path_to_folder}/metadata.json'))
    model_weights = f'{path_to_folder}/model.pt'

    # load model
    model = AutoEncoder(metadata['input_dim'], metadata['hidden_dimension'])

    # save loss information that can be used during anomaly detection
    keys = ["mean_loss", "standard_dev", "percentile_90", "percentile_95", "percentile_98"]
    for key in keys:
        model.reconstruction_loss_metrics[key] = metadata[key]

    model.load_state_dict(torch.load(model_weights))
    print("Model successfully loaded")
    return model


def train_autoencoder_single_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    for x in data_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = mse_loss(output, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


def train_autoencoder(model, train_loader, num_epochs, model_name=None, learning_rate=0.001,
                      weight_decay=0.001, save_path='trained_models/', save_every=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_autoencoder_single_epoch(model, train_loader, optimizer)

        print(f'Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.8f}')

        # Save intermediate checkpoints if requested
        if save_every and epoch % save_every == 0:
            torch.save(model.state_dict(), f'{save_path}/autoencoder_epoch_{epoch}.pt')

    os.makedirs(f'{save_path}/{model_name}/', exist_ok=True)
    torch.save(model.state_dict(), f'{save_path}/{model_name}/model.pt')


def evaluate_autoencoder(model, eval_loader):
    model.eval()
    total_loss = 0
    num_batches = len(eval_loader)

    with torch.no_grad():
        for x in eval_loader:
            output = model(x)
            loss = mse_loss(output, x)
            total_loss += loss.item()

    return total_loss / num_batches

def get_data_mean_squared_errors(ae, dataset):
    observed_loss = []

    ae.eval()
    for batch_data in DataLoader(dataset, batch_size=1):
        reconstructed = ae(batch_data)
        loss = mse_loss(reconstructed, batch_data)
        observed_loss.append(loss.data.item())

    return observed_loss
