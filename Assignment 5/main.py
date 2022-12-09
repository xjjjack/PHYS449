import json

import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse
from pathlib import Path
from PIL import Image

class Data:
    def __init__(self, path, batch_size, n_training_data, n_test_data):
        dataset = np.loadtxt(path, delimiter=' ')
        x_dataset = dataset[:, :-1].astype('float32')
        y_dataset = dataset[:, -1].astype('int64')
        norm_data = x_dataset / 256
        norm_data = np.reshape(norm_data, (len(norm_data), 1, 14, 14))

        x_train = torch.from_numpy(np.reshape(norm_data[:n_training_data], (n_training_data, 1, 14, 14)))
        y_train = torch.from_numpy(y_dataset[:n_training_data])

        x_test = torch.from_numpy(np.reshape(norm_data[n_training_data:], (n_test_data, 1, 14, 14)))
        y_test = torch.from_numpy(y_dataset[n_training_data:])

        train_set = TensorDataset(x_train, y_train)
        test_set = TensorDataset(x_test, y_test)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=False)
        self.train_set = train_loader
        self.test_set = test_loader


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.linear1 = nn.Linear(32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 1, 1))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_epoch(vae, dataloader, loss_function, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader:
        x_hat = vae(x)
        # Evaluate loss
        loss = loss_function(x_hat, x)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()

    return train_loss / len(dataloader.dataset)


# Testing function
def test_epoch(vae, dataloader, loss_function):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():  # No need to track the gradients
        for x, _ in dataloader:
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = loss_function(x_hat, x)
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    parser.add_argument('--param', default='inputs/param.json', help='parameter file name')
    parser.add_argument('--output', '-o', default='result_dir', help='path of results')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--number', '-n', default=10, type=int, help='number of output pdf')
    args = parser.parse_args()
    Path(f'{args.output}').mkdir(parents=True, exist_ok=True)
    # Hyperparameters from json file
    with open(args.param) as paramfile:
        params = json.load(paramfile)
    param = params['data']
    data = Data(param['path'], param['batch_size'], int(param['n_training_data']), int(param['n_test_data']))

    train_loader = data.train_set
    test_loader = data.test_set

    vae = VariationalAutoencoder(latent_dims=4)

    optim = torch.optim.Adam(vae.parameters(), lr=params['exec']['learning_rate'], weight_decay=1e-5)

    loss_function = nn.BCELoss()
    train_loss_lst = []
    test_loss_lst = []
    for epoch in range(int(params['exec']['num_epochs'])):
        train_loss = train_epoch(vae, train_loader, loss_function, optim)
        test_loss = test_epoch(vae, test_loader, loss_function)
        train_loss_lst.append(train_loss)
        test_loss_lst.append(test_loss)
        if args.verbose:
            print('\n EPOCH {}/{} \t train loss {} \t test loss {}'.format(epoch + 1, int(params['exec']['num_epochs']),
                                                                       train_loss, test_loss))

    plt.plot(range(int(params['exec']['num_epochs'])), train_loss_lst, label="Training loss", color="blue")
    plt.plot(range(int(params['exec']['num_epochs'])), test_loss_lst, label="Test loss", color="green")
    plt.legend()
    plt.savefig(Path(args.output) / 'loss.pdf')

    vae.eval()

    with torch.no_grad():
        n = args.number
        index = 1
        while n > 0:
            # sample latent vectors from the normal distribution
            latent = torch.randn(128, 4)

            # reconstruct images from the latent vectors
            img_recon = vae.decoder(latent)
            img_recon = img_recon.cpu()

            for i in range(min(n, 128)):
                img_numpy = img_recon[i].squeeze()
                plt.cla()
                plt.imshow(img_numpy, cmap='gray')
                plt.savefig(Path(args.output) / f"{index}.pdf")
                index += 1

            n -= 128
