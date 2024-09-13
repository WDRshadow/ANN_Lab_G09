import unittest

import numpy as np
from matplotlib import pyplot as plt

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
        mlp.train(X[:-100], Y[:-100], msg=True)
        # test the model and print the accuracy
        mlp.test(X[-100:], Y[-100:])

        # plot the model
        self.plot(mlp)


    def plot(self, model: Module):
        # generate a grid of points and filter them through the model to get the decision boundary whether the point
        # is in class A or B
        x = np.linspace(-2, 3, 100)
        y = np.linspace(-2, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = model.forward(np.array([[X[i, j], Y[i, j]]]))

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




