import json

import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse


class Data:
    def __init__(self, path, batch_size, n_training_data, n_test_data):

        dataset = pd.read_csv(path, delimiter=' ')
        dataset = dataset.astype('float32')

        scaler = MinMaxScaler()
        norm_data = scaler.fit_transform(dataset)
        np.random.shuffle(norm_data)

        x_train = torch.from_numpy(norm_data[:n_training_data, :-1])
        y_train = torch.from_numpy(norm_data[:n_training_data, -1])
        y_train = y_train.to(torch.int64)

        x_test = torch.from_numpy(norm_data[n_training_data:, :-1])
        y_test = torch.from_numpy(norm_data[n_training_data:, -1])
        y_test = y_test.to(torch.int64)

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
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
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
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)

### Training function
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader:
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)

### Testing function
def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)

def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()

def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    parser.add_argument('--param', help='parameter file name')
    parser.add_argument('--res-path', help='path of results')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    data = Data(param['path'], param['batch_size'], int(param['n_training_data']), int(param['n_test_data']))

    train_loader = data.train_set
    test_loader = data.test_set

    print(train_loader[0])

    # d = 4
    #
    # vae = VariationalAutoencoder(latent_dims=d)
    #
    # lr = 1e-3
    #
    # optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    #
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(f'Selected device: {device}')
    #
    # vae.to(device)
    #
    # num_epochs = 50
    #
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(vae, device, train_loader, optim)
    #     val_loss = test_epoch(vae, device, valid_loader)
    #     print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss,
    #                                                                           val_loss))
    #
    # vae.eval()
    #
    # with torch.no_grad():
    #
    #     # sample latent vectors from the normal distribution
    #     latent = torch.randn(128, d, device=device)
    #
    #     # reconstruct images from the latent vectors
    #     img_recon = vae.decoder(latent)
    #     img_recon = img_recon.cpu()
    #
    #     fig, ax = plt.subplots(figsize=(20, 8.5))
    #     show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5))
    #     plt.show()
