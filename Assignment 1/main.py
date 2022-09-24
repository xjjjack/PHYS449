# Write your assignment here
import sys
import os
import json
import numpy as np


def main():
    # load data from .in file
    input_data = np.loadtxt(sys.argv[1])

    # split into 2 matrices, one for x, the other for y (using t by convention of lecture notes)
    phi, t = np.hsplit(input_data, [-1])

    # load json file for parameters
    with open(sys.argv[2]) as f:
        params = json.load(f)

    # Add one column of 1 to the front of Phi for x_0 term
    phi = np.insert(phi, 0, np.ones(np.shape(phi)[0]), 1)

    # write to file
    f = open(os.path.splitext(sys.argv[1])[0] + ".out", "w+")
    try:
        # Analytic solution
        w_analytics = np.matmul(np.matmul(np.linalg.inv(np.matmul(phi.T, phi)), phi.T), t)
        np.savetxt(f, w_analytics, fmt="%.4f", delimiter="\n")
    except np.LinAlgError:
        f.write("Matrix inversion error. Analytic solution is not available.")

    # Gradient descent
    # Initial guess of w as 1's
    w_gds = np.ones(np.shape(phi)[1])
    for k in range(params["num iter"]):
        # create an array to store summation value
        s = np.ones(np.shape(phi)[1])

        # loop through w_gds, calculate summation and store into s
        for i in range(np.shape(phi)[1]):
            for j in range(np.shape(phi)[0]):
                s[i] += - (t[j] - np.dot(w_gds, phi[j])) * phi[j][i]

        # update w_gds
        w_gds = w_gds - params["learning rate"] * s

    f.write("\n")
    if np.isnan(w_gds).any():
        f.write("Divergent solution. Try to use a lower learning rate.")
    else:
        np.savetxt(f, w_gds[:-1], fmt="%.4f", delimiter="\n")
        # write last term separately to avoid newline at the end
        f.write(f"{w_gds[-1]:.4f}")
    f.close()


if __name__ == '__main__':
    main()
