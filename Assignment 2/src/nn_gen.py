import torch
import torch.nn as nn
import torch.nn.functional as func


class Net(nn.Module):

    """
    This nn contains 3fc layers and 2 relu functions.
    Network output dimension is set to 5 to do the multiclass classification task.
    Softmax operation is done in the calculation of Crossentropy so there is no softmax layer in the current network.
    """

    def __init__(self, input_channel, num_class):
        super(Net, self).__init__()
        self.num_class = num_class
        self.fc1 = nn.Linear(input_channel, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_class)

    # Feedforward function
    def forward(self, x):
        h1 = func.relu(self.fc1(x))
        h2 = func.relu(self.fc2(h1))
        y = self.fc3(h2)
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        losses = []
        sample_num = 0
        correct_prediction = 0
        for i, (inputs, label) in enumerate(data):
            optimizer.zero_grad()
            label_hat = self.forward(inputs)
            obj_val = loss(label_hat, label)
            losses.append(obj_val.item())
            obj_val.backward()
            optimizer.step()

            predict = label_hat.argmax(-1)
            cmp = predict.type(label.dtype) == label
            correct_prediction += sum(cmp)
            sample_num += len(label)
        return obj_val.item(), float(correct_prediction/sample_num)

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            for i, (sample, label) in enumerate(data):
                label_hat = self.forward(sample)
                cross_val = loss(label_hat, label)
                predict = label_hat.argmax(-1)
                cmp = predict.type(label.dtype) == label
        return cross_val.item(), float(sum(cmp)/len(cmp))