import json
import os

import torch
from torch.nn.functional import mse_loss


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

    with open(f'{save_path}/{model_name}/metadata.json', "w", encoding="utf-8") as f:
        json.dump(model.metadata, f, indent=4)


def load_autoencoder(path_to_folder):
    from autoencoder_basic import AutoEncoder

    metadata = json.load(open(f'{path_to_folder}/metadata.json'))
    model_weights = f'{path_to_folder}/model.pt'

    model = AutoEncoder(metadata['input_dim'], metadata['hidden_dimension'])

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
