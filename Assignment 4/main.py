import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

def loader(f):
    result = []
    with open(f, "r") as file:
        for row in file:
            r = [1.0 if s == "+" else -1.0 for s in row.strip()]
            result.append(r)
    return np.array(result)


def compute_expectation(a, j):
    return -np.sum(a * np.roll(a, -1) * j)

def compute_set_expectation(m):
    result = []
    for row in m:
        result.append(row * np.roll(row, -1))
    result = np.array(result)
    return np.average(result, axis=0)


def generate_ising(j, size):
    result = []
    s_i = np.random.choice([-1.0, 1.0], len(j))
    result.append(s_i)
    count = 1
    while count < size:
        s_j = np.random.choice([-1.0, 1.0], len(j))
        if compute_expectation(s_j, j) < compute_expectation(s_i, j):
            result.append(s_j)
            s_i = s_j
            count += 1
        else:
            acceptance_threshold = np.exp(compute_expectation(s_i, j) - compute_expectation(s_j, j))
            rand_num = np.random.uniform(0, 1)
            if rand_num < acceptance_threshold:
                result.append(s_j)
                s_i = s_j
                count += 1
    return np.array(result)


def guess(a):
    result = dict()
    for k in range(len(a)):
        if a[k] >= 0:
            result[f"({k}, {(k+1) % len(a)})"] = 1
        else:
            result[f'({k}, {(k+1) % len(a)})'] = -1
    return result


def z(j, size):
    perms = np.array(list(itertools.product([-1, 1], repeat=size)))
    return np.sum(np.exp(np.apply_along_axis(compute_expectation, 1, arr=perms, j=j) * (-1)))


def empirical_p(a, dataset):
    lst = [np.allclose(a,row) for row in dataset]
    return lst.count(True)/len(dataset)


def empirical_dis(dataset, size):
    perms = np.array(list(itertools.product([-1.0, 1.0], repeat=size)))
    return np.apply_along_axis(empirical_p, 1, arr=perms, dataset=dataset)


def KL_divergence(emp_d, j, size):
    acc = 0.0
    perms = np.array(list(itertools.product([-1, 1], repeat=size)))
    Z = z(j, size)
    for k in range(size**2):
        acc += emp_d[k] * np.log(emp_d[k]/(1/Z*np.exp(-compute_expectation(perms[k], j))))
    return acc


if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='Trains an RNN to perform multiplication of binary integers A * B = C')
    parser.add_argument('filename', help='input file', nargs='?')
    parser.add_argument('--compute-expectation', help='compute the expectation value of input dataset')
    parser.add_argument('--compute-empirical', help='compute the empirical distribution of input dataset')
    parser.add_argument('--iteration', default=1000, type=int, help='number of iteration')
    parser.add_argument('--learning-rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--step', default=50, type=int, help='step to show result')
    parser.add_argument('--regularizer', default=1e-2, type=float, help='parameter for L2 regularization')
    parser.add_argument('--res-path', default='results', help='path of results')
    args = parser.parse_args()

    if args.compute_expectation:
        data = loader(args.compute_expectation)
        print(compute_set_expectation(data))
    elif args.compute_empirical:
        data = loader(args.compute_empirical)
        print(empirical_dis(data, len(data[0])))
    else:
        data = loader(args.filename)
        width = len(data[0])
        length = len(data)
        empirical_distribution = empirical_dis(data, width)
        j_guess = np.random.choice([-1.0, 1.0], width)
        divergence = []
        for i in range(args.iteration):
            generated_data = generate_ising(j_guess, length)
            grad = compute_set_expectation(data) - compute_set_expectation(generated_data) - args.regularizer * j_guess
            j_guess += args.learning_rate * grad
            divergence.append(KL_divergence(empirical_distribution, j_guess, width))
            if i % args.step == 0:
                print(j_guess)
        my_guess = guess(j_guess)
        s = '{'
        for key, value in my_guess.items():
            s += key + ': ' + str(value) + ', '
        s = s[:-2]
        s += '}'
        print(s)
        np.savetxt(Path(args.res_path) / 'divergence.txt', divergence)
        plt.plot(range(args.iteration), divergence, label="KL divergence", color="blue")
        plt.legend()
        plt.savefig(Path(args.res_path) / 'fig.pdf')


