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
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# data_dir = 'dataset'
#
# train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
# test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)
#
# train_transform = transforms.Compose([
# transforms.ToTensor(),
# ])
#
# test_transform = transforms.Compose([
# transforms.ToTensor(),
# ])
#
# train_dataset.transform = train_transform
# test_dataset.transform = test_transform
#
# m=len(train_dataset)
#
# train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
# batch_size=256
#
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
# valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

# print(train_dataset[0])
with open('data/even_mnist.csv', 'r') as file:
    lines = file.readlines()
    idx = np.random.randint(0, len(lines))
    img = np.array(lines[idx].split()[:-1]).astype('float32')
    img = img / 255
    img = np.reshape(img, (14,14))
    label = lines[idx].split()[-1]
    print(img)
    print(label)
    plt.imshow(img, cmap='gray')
    plt.show()

# img, label = train_data[0]
# print(img)
# print(img.dtype)
# print(label)
# # plt.imshow(img.squeeze(), cmap='gray')
# # plt.show()