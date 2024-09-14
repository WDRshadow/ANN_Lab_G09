import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import part2
from utils import Module, Layer, ReLU, Tanh, DataGenerator2, GaussFunctionData


class MLP(Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 5, study_rate=0.001, epochs=3000):
        super().__init__(study_rate, epochs)
        self.layer1 = Layer(input_dim, hidden_dim, ReLU)
        self.layer2 = Layer(hidden_dim, output_dim, Tanh)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)


class MLP2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 5):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.Tanh(self.fc2(x))
        return x


def plot_loss(losses, add_title=''):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs' + add_title)
    plt.show()


def plot_losses(losses, add_title='', label=''):
    for i, loss in enumerate(losses):
        plt.plot(loss, label=label + str(i + 1))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs' + add_title)
    plt.show()


def plot(data_generator: DataGenerator2, model, add_title=''):
    # generate a grid of points and filter them through the model to get the decision boundary whether the point
    # is in class A or B
    x = np.linspace(-2, 3, 100)
    y = np.linspace(-2, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if model.__class__.__name__ == 'MLP':
                Z[i, j] = model(np.array([[X[i, j], Y[i, j]]]))
            elif model.__class__.__name__ == 'MLP2':
                Z[i, j] = model(torch.tensor([[X[i, j], Y[i, j]]]).float()).item()

    # plot the decision boundary
    plt.contourf(X, Y, Z, levels=1, colors=['blue', 'red'], alpha=0.5)

    labels = data_generator.data[1].flatten()
    classA = data_generator.data[0][labels == 1]
    classB = data_generator.data[0][labels == -1]

    plt.scatter(classA.transpose()[0, :], classA.transpose()[1, :], color='red', label='Class A')
    plt.scatter(classB.transpose()[0, :], classB.transpose()[1, :], color='blue', label='Class B')

    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Decision Boundary" + add_title)
    plt.legend()
    plt.show()


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.study_rate = 0.001
        self.epochs = 500

        self.data_generator = DataGenerator2(
            n=300,
            mA=[1.5, 0.5],
            sigmaA=0.2,
            mB=[-0.0, -0.1],
            sigmaB=0.3,
            mA2=[-1.5, 0.3],
            sigmaA2=0.2
        )

    def train_and_test(self, X_val, Y_val, add_title='', model: Module = None, is_plot=True, is_plot_loss=True):
        X, Y = self.data_generator.data
        if model is None:
            model = MLP(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        losses = model.train(X, Y)
        model.test(X_val, Y_val)
        if is_plot:
            plot(self.data_generator, model, add_title)
        if is_plot_loss:
            plot_loss(losses, add_title)
        return losses

    def test(self):
        X_val, Y_val = self.data_generator.randomly_remove_data(0.3, 0.3)
        self.train_and_test(X_val, Y_val)

    def test_pytorch(self):
        X, Y = self.data_generator.data
        X_train = X[:-100]
        Y_train = Y[:-100].T[0]
        X_val = X[-100:]
        Y_val = Y[-100:].T[0]
        mlp = MLP2(2, 1)
        losses = part2.train(mlp, torch.tensor(X_train).float(), torch.tensor(Y_train).float(),
                             torch.tensor(X_val).float(),
                             torch.tensor(Y_val).float(), study_rate=self.study_rate, epochs=self.epochs)
        part2.test_model(mlp, torch.tensor(X_val).float(), torch.tensor(Y_val).float())

        # plot the model
        plot(self.data_generator, mlp, ' - PyTorch')
        plot_loss(losses, ' - PyTorch')

    def test_remove_case(self):
        losses = []
        # randomly remove 25% of the data from each class
        X_val, Y_val = self.data_generator.randomly_remove_data(0.25, 0.25)
        losses.append(self.train_and_test(X_val, Y_val, ' - 25% of data removed from each class', is_plot_loss=False))
        # randomly remove 50% of the data from A class
        X_val, Y_val = self.data_generator.randomly_remove_data(0.5, 0)
        losses.append(self.train_and_test(X_val, Y_val, ' - 50% of data removed from class A', is_plot_loss=False))
        # Randomly remove 20% of the data points from class A where x0 < 0 and 80% of the data points from class A where x0 >= 0.
        X_val, Y_val = self.data_generator.randomly_remove_specific_data()
        losses.append(
            self.train_and_test(X_val, Y_val, ' - 20% of data removed from class A where x0 < 0 and 80% where x0 >= 0',
                                is_plot_loss=False))
        plot_losses(losses, ' - Remove Data Case', 'Case = ')

    def test_hidden_dim_case(self):
        losses = []
        X_val, Y_val = self.data_generator.randomly_remove_data(0.3, 0.3)
        for hidden_dim in range(1, 7):
            mlp = MLP(2, 1, hidden_dim=hidden_dim, study_rate=self.study_rate, epochs=self.epochs)
            losses.append(self.train_and_test(X_val, Y_val, ' - hidden_dim = ' + str(hidden_dim), mlp, False, False))
        plot_losses(losses, ' - Hidden Dimension', 'hidden_dim = ')

    def test_gauss(self):
        self.data_generator = GaussFunctionData()
        self.data_generator.plot()
        X_val, Y_val = self.data_generator.randomly_remove_data(0.3)
        self.data_generator.reset_data()
        mlp = MLP(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        self.train_and_test(X_val, Y_val, ' - Gauss Function Data', mlp, False)
        pred = mlp(self.data_generator.data_copy[0])
        self.data_generator.Z = pred.reshape(self.data_generator.Z.shape)
        self.data_generator.plot()
