import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Data:
    def __init__(self, path, batch_size, n_training_data, n_test_data):

        dataset = pd.read_csv(path, delimiter=' ')
        dataset = dataset.astype('float32')

        scaler = MinMaxScaler()
        norm_data = scaler.fit_transform(dataset)
        np.random.shuffle(norm_data)

        x_train = torch.from_numpy(norm_data[:n_training_data, :-1])
        y_train = torch.from_numpy(norm_data[:n_training_data, -1])
        # x_train = np.array(x_train, dtype=np.float32)
        y_train = y_train.to(torch.int64)

        x_test = torch.from_numpy(norm_data[n_training_data:, :-1])
        y_test = torch.from_numpy(norm_data[n_training_data:, -1])
        # x_test = np.array(x_test, dtype=np.float32)
        y_test = y_test.to(torch.int64)

        train_set = TensorDataset(x_train, y_train)
        test_set = TensorDataset(x_test, y_test)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=False)
        self.train_set = train_loader
        self.test_set = test_loader
