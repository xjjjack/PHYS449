import json, argparse, torch, sys
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sys.path.append('src')
from nn_gen import Net
from data_gen import Data

def plot_results(obj_vals, cross_vals, train_acc, test_acc):
    assert len(obj_vals) == len(cross_vals), 'Length mismatch between the curves'
    assert len(train_acc) == len(test_acc), 'Length mismatch between the curves'
    num_epochs = len(obj_vals)

    # Plot saved in results folder
    plt.plot(range(num_epochs), obj_vals, label="Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label="Test loss", color="green")
    plt.plot(range(num_epochs), train_acc, label="Training Accuracy", color="red")
    plt.plot(range(num_epochs), test_acc, label="Test Accuracy", color="black")
    plt.legend()
    plt.savefig(args.res_path + '/fig.pdf')
    # plt.close()


def prep_demo(param):
    # Construct a model and dataset
    model = Net(param['input_channel'], param['num_class'])
    data = Data(param['path'], param['batch_size'], int(param['n_training_data']), int(param['n_test_data']))
    return model, data


def run_demo(param, model, data):

    # Define an optimizer and the loss function
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    obj_vals = []
    cross_vals = []
    train_acc = []
    test_acc = []
    num_epochs = int(param['num_epochs'])

    # Training loop
    for epoch in range(num_epochs):

        train_val, acc1 = model.backprop(data.train_set, loss, epoch, optimizer)
        obj_vals.append(train_val)
        train_acc.append(acc1)

        test_val, acc2 = model.test(data.test_set, loss, epoch)
        cross_vals.append(test_val)
        test_acc.append(acc2)

        if (epoch+1) % param['display_epochs'] == 0:
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs) + \
                    '\t Training Loss: {:.4f}'.format(train_val) + \
                    '\t Test Loss: {:.4f}'.format(test_val) + \
                    '\t Train Accuracy: {:.4f}'.format(acc1) + \
                    '\t Test Accuracy: {:.4f}'.format(acc2))
    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))
    print('Final training Accuracy: {:.4f}'.format(train_acc[-1]))
    print('Final test Accuracy: {:.4f}'.format(test_acc[-1]))

    return obj_vals, cross_vals, train_acc, test_acc


if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    parser.add_argument('--param', help='parameter file name')
    parser.add_argument('--res-path', help='path of results')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    model, data = prep_demo(param['data'])
    obj_vals, cross_vals, train_acc, test_acc = run_demo(param['exec'], model, data)
    plot_results(obj_vals, cross_vals, train_acc, test_acc)