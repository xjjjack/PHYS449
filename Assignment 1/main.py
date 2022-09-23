# Write your assignment here
import sys
import os
import json
import numpy as np


def main():

    # load data from .in file
    input_data = np.loadtxt(sys.argv[1])
    print(input_data)
    # split into 2 matrices, one for x, the other for y (using t by convention of lecture notes)
    phi, t = np.hsplit(input_data, [-1])
    with open(sys.argv[2]) as f:
        params = json.load(f)

    # Add one column of 1 to the front of Phi for x_0 term
    phi = np.insert(phi, 0, np.ones(np.shape(phi)[0]), 1)

    print(t)
    print(phi)
    print(np.shape(phi))

    # Analytic solution
    w_analytics = np.matmul(np.matmul(np.linalg.inv(np.matmul(phi.T, phi)), phi.T), t)
    print(w_analytics)

    # Gradient descent
    # Initial guess of w
    w_gds = np.ones(np.shape(phi)[1])
    print(np.shape(np.sum(w_gds*phi, 1)))

    for i in range(params["num iter"]):
        for j in range(np.shape(phi)[1]):
            s = 0.0
            for k in range(np.shape(phi)[0]):
                s += - (t[k] - np.dot(w_gds, phi[k])) * phi[k][j]
            w_gds[j] = w_gds[j] - params["learning rate"] * s

    print(w_gds)


if __name__ == '__main__':
    main()
