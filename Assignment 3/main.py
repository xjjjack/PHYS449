import argparse, json
import numpy as np
from rnn import SimpleRnn
from pathlib import Path
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='Trains an RNN to perform multiplication of binary integers A * B = C')
    parser.add_argument('--param', default='param/param.json', metavar='param.json', help='file containing hyperparameters')
    parser.add_argument('--train-size', default=1000, type=int, help='size of the generated training set')
    parser.add_argument('--test-size', default=100, type=int, help='size of the generated test set')
    parser.add_argument('--seed', default=8888, type=int, help='random seed used for creating the datasets')
    parser.add_argument('--res-path', default='results', help='path of results')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    rng = np.random.default_rng(seed=args.seed)
    random_array_train_A = rng.integers(low=0, high=2, size=(args.train_size, 8))
    np.savetxt(Path(args.res_path) / 'train_A.txt', random_array_train_A, fmt='%d', delimiter=' ')
    random_array_train_B = rng.integers(low=0, high=2, size=(args.train_size, 8))
    np.savetxt(Path(args.res_path) / 'train_B.txt', random_array_train_B, fmt='%d', delimiter=' ')
    a = random_array_train_A.dot(1 << np.arange(random_array_train_A.shape[-1]))
    b = random_array_train_B.dot(1 << np.arange(random_array_train_B.shape[-1]))
    c = a * b
    random_array_train_C = ((c[:, None] & (1 << np.arange(16))) > 0).astype(int)
    np.savetxt(Path(args.res_path) / 'train_C.txt', random_array_train_C, fmt='%d', delimiter=' ')

    random_array_test_A = rng.integers(low=0, high=2, size=(args.test_size, 8))
    np.savetxt(Path(args.res_path) / 'test_A.txt', random_array_test_A, fmt='%d', delimiter=' ')
    random_array_test_B = rng.integers(low=0, high=2, size=(args.test_size, 8))
    np.savetxt(Path(args.res_path) / 'test_B.txt', random_array_test_B, fmt='%d', delimiter=' ')
    a = random_array_test_A.dot(1 << np.arange(random_array_test_A.shape[-1]))
    b = random_array_test_B.dot(1 << np.arange(random_array_test_B.shape[-1]))
    c = a * b
    random_array_test_C = ((c[:, None] & (1 << np.arange(16))) > 0).astype(int)
    np.savetxt(Path(args.res_path) / 'test_C.txt', random_array_test_C, fmt='%d', delimiter=' ')

    net1 = SimpleRnn(param['model']['scale'], param['model']['input_dim'],
                    param['model']['hidden_dim'], param['model']['output_dim'], param['model']['alpha'])

    trainset = {'a': random_array_train_A.tolist(), 'b': random_array_train_B.tolist(), 'c': random_array_train_C}
    testset = {'a': random_array_test_A.tolist(), 'b': random_array_test_B.tolist(), 'c': random_array_test_C}

    train_loss, test_loss = net1.train(trainset, testset, args.train_size, args.test_size, epoch=param['optim']['epoch'], interval=param['optim']['report_interval'])
    print('----------Swap----------')
    net2 = SimpleRnn(param['model']['scale'], param['model']['input_dim'],
                     param['model']['hidden_dim'], param['model']['output_dim'], param['model']['alpha'])

    trainset_swap = {'b': random_array_train_B.tolist(), 'a': random_array_train_A.tolist(), 'c': random_array_train_C}
    testset_swap = {'b': random_array_test_B.tolist(), 'a': random_array_test_A.tolist(), 'c': random_array_test_C}

    train_loss_swap, test_loss_swap = net2.train(trainset_swap, testset_swap, args.train_size, args.test_size, epoch=param['optim']['epoch'], interval=param['optim']['report_interval'])
    # Plot saved in results folder
    plt.plot(range(0, param['optim']['epoch'], param['optim']['report_interval'])[1:], train_loss, label="Training loss", color="blue")
    plt.plot(range(0, param['optim']['epoch'], param['optim']['report_interval'])[1:], test_loss, label="Test loss", color="green")
    plt.plot(range(0, param['optim']['epoch'], param['optim']['report_interval'])[1:], train_loss_swap, label="Training loss after swap", color="blue")
    plt.plot(range(0, param['optim']['epoch'], param['optim']['report_interval'])[1:], test_loss_swap, label="Test loss after swap", color="green")
    plt.legend()
    plt.savefig(Path(args.res_path) / 'fig.pdf')

