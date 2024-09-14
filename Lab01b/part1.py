import unittest

import numpy as np
from matplotlib import pyplot as plt

from utils import MLP, Layer, ReLU, Tanh, DataGenerator2, GaussFunctionData


class DLP(MLP):
    """
    Double Layer Perceptron, which has 2 layers, the first layer has hidden_dim neurons and the second layer has
    output_dim neurons. The activation function of the first layer is ReLU and the second layer is Tanh.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 5, study_rate=0.001, epochs=3000):
        super().__init__(study_rate, epochs)
        self.layers.append(Layer(input_dim, hidden_dim, ReLU))
        self.layers.append(Layer(hidden_dim, output_dim, Tanh))


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
            Z[i, j] = model(np.array([[X[i, j], Y[i, j]]]))

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

    def train_and_test(self, X_val, Y_val, add_title='', model: MLP = None, is_plot=True, is_plot_loss=True):
        X, Y = self.data_generator.data
        if model is None:
            model = DLP(2, 1, study_rate=self.study_rate, epochs=self.epochs)
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
            mlp = DLP(2, 1, hidden_dim=hidden_dim, study_rate=self.study_rate, epochs=self.epochs)
            losses.append(self.train_and_test(X_val, Y_val, ' - hidden_dim = ' + str(hidden_dim), mlp, False, False))
        plot_losses(losses, ' - Hidden Dimension', 'hidden_dim = ')

    def test_gauss(self):
        self.data_generator = GaussFunctionData()
        self.data_generator.plot()
        X_val, Y_val = self.data_generator.randomly_remove_data(0.3)
        self.data_generator.reset_data()
        mlp = DLP(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        self.train_and_test(X_val, Y_val, ' - Gauss Function Data', mlp, False)
        pred = mlp(self.data_generator.data_copy[0])
        self.data_generator.Z = pred.reshape(self.data_generator.Z.shape)
        self.data_generator.plot()

    def test_encoder(self):
        # randomly generate data with 1 dimension in range 0-7 with type int
        X = np.random.randint(0, 7, (300, 1))
        X_val = np.random.randint(0, 7, (30, 1))

        def one_hot_encode(x):
            y = np.zeros((x.size, 8)) - 1
            y[np.arange(x.size), x.flatten()] = 1
            return y

        X = one_hot_encode(X)
        X_val = one_hot_encode(X_val)
        encoder = MLP(0.001, 5000)
        encoder.layers.append(Layer(8, 3, Tanh))
        encoder.layers.append(Layer(3, 8, Tanh))
        losses = encoder.train(X, X)
        encoder.test(X_val, X_val)
        plot_loss(losses, ' - Encoder')
