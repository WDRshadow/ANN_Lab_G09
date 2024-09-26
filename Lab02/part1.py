import unittest

import matplotlib.pyplot as plt
import numpy as np

from utils import RBF, One_Dim_Function, Read_Files


def plot_two_lines(X, Y_train, Y_pred, plot_title=''):
    plt.plot(X, Y_train, label='Ground Truth')
    plt.plot(X, Y_pred, label='Prediction')
    plt.legend()
    plt.title(plot_title)
    plt.show()


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.samples = 100
        self.data_generator_sin2x = One_Dim_Function(0, 2 * np.pi, self.samples, function=lambda x: np.sin(2 * x))
        self.data_generator_square2x = One_Dim_Function(-50, 50, self.samples, function=lambda x: (2 * x) ** 2)
        self.ballist: np.ndarray = Read_Files("data_lab2/ballist.dat").read_matrix(" ", "\t", float)
        self.balltest: np.ndarray = Read_Files("data_lab2/balltest.dat").read_matrix(" ", "\t", float)

    @staticmethod
    def _test(data_generator: One_Dim_Function, model: RBF, rbf_unit=10, sigma=1.0, competitive_learning=False,
              is_plot=True, plot_title=None):
        X_train, y_train = data_generator.data
        if competitive_learning:
            model.train(X_train, y_train)
        else:
            centers = np.linspace(data_generator.data[0][0], data_generator.data[0][-1], rbf_unit).reshape(-1, 1)
            model.set_C_n_Sigma(centers, np.array([sigma] * rbf_unit))
            model.backward(X_train, y_train)
        model.test(data_generator.data[0], data_generator.data[1])
        if competitive_learning:
            data_generator.reset_data()
        if is_plot:
            y_pred = model(data_generator.data[0])
            plot_two_lines(data_generator.data[0], data_generator.data[1], y_pred, plot_title)
        data_generator.reset_data()

    def test_sin_2x(self):
        rbf_unit, sigma = 10, 1.0
        rbf_net = RBF(input_dim=1, rbf_dim=rbf_unit)
        self._test(self.data_generator_sin2x, rbf_net, rbf_unit, sigma, plot_title='sin(2x)')

    def test_sin_2x_gaussian(self):
        rbf_unit, sigma = 10, 1.0
        self.data_generator_sin2x.add_gaussian_noise(0, 0.1)
        rbf_net = RBF(input_dim=1, rbf_dim=rbf_unit)
        self._test(self.data_generator_sin2x, rbf_net, rbf_unit, sigma, plot_title='sin(2x) with Gaussian noise')

    def test_sin_2x_competitive_learning(self):
        rbf_unit, study_rate, epochs = 10, 0.001, 1000
        self.data_generator_square2x.randomly_mix_data()
        rbf_net = RBF(input_dim=1, rbf_dim=rbf_unit)
        self._test(self.data_generator_sin2x, rbf_net, competitive_learning=True, plot_title='sin(2x) with Competitive Learning')

    def test_square_2x(self):
        rbf_unit, sigma = 50, 2.0
        rbf_net = RBF(input_dim=1, rbf_dim=rbf_unit)
        self._test(self.data_generator_square2x, rbf_net, rbf_unit, sigma, plot_title='2x^2')

    def test_square_2x_gaussian(self):
        rbf_unit, sigma = 50, 2.0
        self.data_generator_square2x.add_gaussian_noise(0, 100)
        rbf_net = RBF(input_dim=1, rbf_dim=rbf_unit)
        self._test(self.data_generator_square2x, rbf_net, rbf_unit, sigma, plot_title='2x^2 with Gaussian noise')

    def test_square_2x_competitive_learning(self):
        rbf_unit, study_rate, epochs = 100, 0.001, 10000
        self.data_generator_square2x.randomly_mix_data()
        rbf_net = RBF(input_dim=1, rbf_dim=rbf_unit, study_rate=study_rate, epochs=epochs)
        # set up random initial C (scale up to 100)
        rbf_net.layers[0].C = rbf_net.layers[0].C * 100
        # set initial sigma to 2.0
        rbf_net.layers[0].Sigma = np.array([2.0] * rbf_unit)
        self._test(self.data_generator_square2x, rbf_net, competitive_learning=True, plot_title='2x^2 with Competitive Learning')

    def test_2_dim_function(self):
        rbf_unit, sigma = 10, 1.0
        X_train = self.ballist[:, 0:2]
        y_train = self.ballist[:, 2:4]
        X_test = self.balltest[:, 0:2]
        y_test = self.balltest[:, 2:4]
        rbf_net = RBF(input_dim=2, rbf_dim=rbf_unit)
        rbf_net.train(X_train, y_train)
        rbf_net.test(X_test, y_test)
        y_pred = rbf_net(X_test)
        plot_two_lines(np.arange(y_test.shape[0]), y_test[:, 0], y_pred[:, 0], plot_title='Ballist Distance')
        plot_two_lines(np.arange(y_test.shape[0]), y_test[:, 1], y_pred[:, 1], plot_title='Ballist Height')
