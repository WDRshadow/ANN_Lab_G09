import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import part2
from DataGenerator import DataGenerator2
from Module import Module, Layer, Sigmoid, ReLU


class MLP(Module):
    def __init__(self, input_dim: int, output_dim: int, study_rate=0.001, epochs=3000):
        super().__init__(study_rate, epochs)
        self.layer1 = Layer(input_dim, 10, ReLU)
        self.layer2 = Layer(10, 10, ReLU)
        self.layer3 = Layer(10, 5, Sigmoid)
        self.layer4 = Layer(5, output_dim, Sigmoid)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)


class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, 1)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.Sigmoid(self.fc3(x))
        x = self.Sigmoid(self.fc4(x))
        return x


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.study_rate = 0.001
        self.epochs = 30000

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
        mlp = MLP(2, 1, self.study_rate, self.epochs)
        X, Y = self.data_generator.data
        # 20% of the data is used for testing
        losses = mlp.train(X[:-100], Y[:-100], msg=True)
        # test the model and print the accuracy
        mlp.test(X[-100:], Y[-100:])

        # plot the model
        self.plot(mlp)

        # plot the loss
        self.plot_loss(losses)

    def test_third_lib(self):
        X, Y = self.data_generator.data
        X_train = torch.tensor(X[:-100]).float()
        Y_train = torch.tensor(Y[:-100]).float()
        X_val = torch.tensor(X[-100:]).float()
        Y_val = torch.tensor(Y[-100:]).float()
        mlp = MLP2()
        part2.train(mlp, X_train, Y_train, X_val, Y_val, study_rate=self.study_rate, epochs=self.epochs)
        predictions = part2.test_model(mlp, X_val)
        accuracy = 1 - np.mean(np.abs(predictions - Y_val.numpy()))
        print(f'Accuracy: {accuracy}')

        # plot the model
        self.plot(mlp)


    @staticmethod
    def plot_loss(losses):
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.show()


    def plot(self, model):
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

        labels = self.data_generator.data[1].flatten()
        classA = self.data_generator.data[0][labels == 1]
        classB = self.data_generator.data[0][labels == -1]

        plt.scatter(classA.transpose()[0, :], classA.transpose()[1, :], color='red', label='Class A')
        plt.scatter(classB.transpose()[0, :], classB.transpose()[1, :], color='blue', label='Class B')

        plt.xlim(-2, 3)
        plt.ylim(-2, 3)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title("Decision Boundary")
        plt.legend()
        plt.show()




