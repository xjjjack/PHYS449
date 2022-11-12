import numpy as np
import copy, random
from utils import sigmoid, sigmoid_output_to_derivative as grad
from utils import one_hot, softmax, softmax2bit, unpackbits


class SimpleRnn:
    def __init__(self, size, input_dim, hidden_dim, output_dim, alpha):

        # initialize neural network parameters
        self.input_layer = 2 * np.random.random((input_dim, hidden_dim)) - 1
        self.output_layer = 2 * np.random.random((hidden_dim, output_dim)) - 1
        self.hidden_layer = 2 * np.random.random((hidden_dim, hidden_dim)) - 1
        self.input_update = np.zeros_like(self.input_layer)
        self.output_update = np.zeros_like(self.output_layer)
        self.hidden_update = np.zeros_like(self.hidden_layer)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha              # learning rate

        # generate training dataset
        self.size = size                # max length of outputs
        self.largest = pow(2, size)     # largest value of binary multiplication
        self.int2binary = {}
        binary = unpackbits(size)
        for i in range(self.largest):
            self.int2binary[i] = binary[i]

    def train(self, trainset, testset, train_size, test_size, epoch, interval):
        # training process
        loss = 0
        train_loss = []
        test_loss = []
        for j in range(epoch):
            for i in range(train_size):
                # pick a training sample (a * b = c)
                a = trainset['a'][i]
                b = trainset['b'][i]
                if len(a) < self.size:
                    for k in range(self.size - len(a)):
                        a.insert(0, 0)
                        b.insert(0, 0)
                a = np.array(a)
                b = np.array(b)

                # true answer
                c = np.array(trainset['c'][i])

                # network output in binary encoding
                d = np.zeros_like(c)

                overallError = 0
                layer_2_deltas = list()
                layer_1_values = list()
                layer_1_values.append(np.zeros(self.hidden_dim))

                # moving along the positions in the binary encoding
                for position in range(self.size):

                    x = np.array([one_hot(a[self.size - position - 1]), one_hot(b[self.size - position - 1])]).reshape(-1)
                    y = np.array([one_hot(c[self.size - position - 1])])

                    # hidden layer (input ~+ prev_hidden)
                    layer_1 = sigmoid(np.dot(x, self.input_layer) + np.dot(layer_1_values[-1], self.hidden_layer))

                    # output layer (new binary representation)
                    layer_2 = sigmoid(np.dot(layer_1, self.output_layer))

                    # compute error of output
                    layer_2_error = y - layer_2
                    layer_2_deltas.append(layer_2_error * grad(layer_2))
                    overallError += np.linalg.norm(layer_2_error)

                    # decode estimate
                    d[self.size - position - 1] = softmax2bit(softmax(layer_2))

                    # store hidden layer
                    layer_1_values.append(copy.deepcopy(layer_1))

                future_layer_1_delta = np.zeros(self.hidden_dim)
                loss += np.linalg.norm(np.array(c) - np.array(d))

                for position in range(self.size):
                    x = np.array([one_hot(a[position]), one_hot(b[position])]).reshape(-1)
                    layer_1 = layer_1_values[-position - 1]
                    prev_layer_1 = layer_1_values[-position - 2]

                    # error at output layer
                    layer_2_delta = layer_2_deltas[-position - 1]
                    # error at hidden layer
                    layer_1_delta = (future_layer_1_delta.dot(self.hidden_layer.T) +
                                     layer_2_delta.dot(self.output_layer.T)) * grad(layer_1)
                    # calculate gradient for weighted matrices
                    self.output_update += np.outer(layer_1, layer_2_delta)
                    self.hidden_update += np.outer(prev_layer_1, layer_1_delta)
                    self.input_update += np.outer(x.T, layer_1_delta)

                    future_layer_1_delta = layer_1_delta

                # update weighted matrices
                self.input_layer += self.input_update * self.alpha
                self.output_layer += self.output_update * self.alpha
                self.hidden_layer += self.hidden_update * self.alpha

                self.input_update *= 0
                self.output_update *= 0
                self.hidden_update *= 0


            # print out results
            if j % interval == 0 and j > 0:
                # print("Error: %s" % str(overallError))
                # print("Pred: %s" % str(d))
                # print("True: %s" % str(c))
                # out = 0
                # for index, x in enumerate(reversed(d)):
                #     out += x * pow(2, index)
                train_loss.append(loss/train_size)
                print(f"Train loss: {loss/train_size}")
                test_loss.append(self.test(testset, test_size))
                print(f"Test loss: {self.test(testset, test_size)}")
                print("------------")
            loss = 0
        return train_loss, test_loss

    def test(self, testset, test_size):
        loss = 0
        for i in range(test_size):
            a = testset['a'][i]
            b = testset['b'][i]
            if len(a) < self.size:
                for k in range(self.size - len(a)):
                    a.insert(0, 0)
                    b.insert(0, 0)
            c = testset['c'][i]
            d = self.forward(a, b)
            loss += np.linalg.norm(np.array(c)-np.array(d))
        return loss/test_size

    def forward(self, a, b):
        # a, b, and result are encoded in "little endian"
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.hidden_dim))
        result = np.zeros(self.size, dtype=int).tolist()

        # align inputs
        if len(a) < self.size:
            for i in range(self.size - len(a)):
                a.insert(0, 0)
        if len(b) < self.size:
            for i in range(self.size - len(b)):
                b.insert(0, 0)

        # moving along the positions in the binary encoding
        for position in range(self.size):
            x = np.array([one_hot(a[position]), one_hot(b[position])]).reshape(-1)
            layer_1 = sigmoid(np.dot(x, self.input_layer) + np.dot(layer_1_values[-1], self.hidden_layer))
            layer_2 = sigmoid(np.dot(layer_1, self.output_layer))
            result[position] = softmax2bit(softmax(layer_2))
            layer_1_values.append(copy.deepcopy(layer_1))
        return result


