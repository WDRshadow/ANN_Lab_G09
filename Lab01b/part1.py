import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import part2
from utils import Module, Layer, ReLU, Tanh, DataGenerator2


class MLP(Module):
    def __init__(self, input_dim: int, output_dim: int, study_rate=0.001, epochs=3000):
        super().__init__(study_rate, epochs)
        self.layer1 = Layer(input_dim, 10, ReLU)
        self.layer2 = Layer(10, 10, ReLU)
        self.layer3 = Layer(10, 5, Tanh)
        self.layer4 = Layer(5, output_dim, Tanh)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)


class MLP2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, output_dim)
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.Tanh(self.fc3(x))
        x = self.Tanh(self.fc4(x))
        return x


def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()


def plot(data_generator: DataGenerator2, model):
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
    plt.title("Decision Boundary")
    plt.legend()
    plt.show()


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.study_rate = 0.001
        self.epochs = 3000

        self.data_generator = DataGenerator2(
            n=300,
            mA=[1.5, 0.5],
            sigmaA=0.2,
            mB=[-0.0, -0.1],
            sigmaB=0.3,
            mA2=[-1.5, 0.3],
            sigmaA2=0.2
        )

    def test(self):
        X, Y = self.data_generator.data
        mlp = MLP(2, 1, self.study_rate, self.epochs)
        # 20% of the data is used for testing
        losses = mlp.train(X[:-100], Y[:-100], msg=True)
        # test the model and print the accuracy
        mlp.test(X[-100:], Y[-100:])

        # plot the model
        plot(self.data_generator, mlp)
        plot_loss(losses)

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
        plot(self.data_generator, mlp)
        plot_loss(losses)

    def test_all(self):
        self.test()
        self.test_pytorch()
