import numpy as np


# compute sigmoid function
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# compute softmax function
def softmax(x):
    output = np.exp(x)
    output /= np.sum(output)
    return output


# compute derivative of sigmoid function
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# one-hot encoding for binary multiplication
def one_hot(n):
    if n == 0:
        return [0, 1]
    else:
        return [1, 0]


# turn softmax result to 0 or 1
def softmax2bit(x):
    if x[0] > x[1]:
        return 1
    else:
        return 0


# realization of np.unpackbits func
def unpackbits(size):
    n = pow(2, size)
    output = np.zeros((n, size), dtype=np.int)
    for i in range(n):
        binary = bin(i)[2:]
        if len(binary) < size:
            for k in range(size-len(binary)):
                binary = '0' + binary
        for j in range(size):
            output[i][j] = int(binary[j])
    return output

