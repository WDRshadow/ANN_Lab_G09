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
        self.data_generator_square2xbox = One_Dim_Function(0, 2 * np.pi, self.samples, function=lambda x: np.where(np.sin(x) >= 0, 1, -1))

        # self.ballist: np.ndarray = Read_Files("data_lab2/ballist.dat").read_matrix(" ", "\t", float)
        # self.balltest: np.ndarray = Read_Files("data_lab2/balltest.dat").read_matrix(" ", "\t", float)

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

    def test_rbf_residuals_vs_units(self):
        rbf_units_range = range(1, 21)
        
        residuals_sin2x = []
        residuals_square2xbox = []
        residuals_sin2x_noise = []
        residuals_square2xbox_noise = []

        sin2x_gen = self.data_generator_sin2x
        square2xbox_gen = self.data_generator_square2xbox
        
        sin2x_noise_gen = One_Dim_Function(0, 2 * np.pi, self.samples, function=lambda x: np.sin(2 * x))
        square2xbox_noise_gen = One_Dim_Function(0, 2 * np.pi, self.samples, function=lambda x: np.where(np.sin(x) >= 0, 1, -1))
        sin2x_noise_gen.add_gaussian_noise(0, 0.5)
        square2xbox_noise_gen.add_gaussian_noise(0, 0.5)

        for rbf_unit in rbf_units_range:
            rbf_sin2x = RBF(input_dim=1, rbf_dim=rbf_unit)
            rbf_square2xbox = RBF(input_dim=1, rbf_dim=rbf_unit)
            rbf_sin2x_noise = RBF(input_dim=1, rbf_dim=rbf_unit)
            rbf_square2xbox_noise = RBF(input_dim=1, rbf_dim=rbf_unit)
            
            X_train_sin2x, y_train_sin2x = sin2x_gen.data
            X_train_square2xbox, y_train_square2xbox = square2xbox_gen.data
            centers_sin2x = np.linspace(X_train_sin2x[0], X_train_sin2x[-1], rbf_unit).reshape(-1, 1)
            centers_square2xbox = np.linspace(X_train_square2xbox[0], X_train_square2xbox[-1], rbf_unit).reshape(-1, 1)

            rbf_sin2x.set_C_n_Sigma(centers_sin2x, np.array([1.0] * rbf_unit))
            rbf_square2xbox.set_C_n_Sigma(centers_square2xbox, np.array([1.0] * rbf_unit))
            rbf_sin2x_noise.set_C_n_Sigma(centers_sin2x, np.array([1.0] * rbf_unit))
            rbf_square2xbox_noise.set_C_n_Sigma(centers_square2xbox, np.array([1.0] * rbf_unit))

            rbf_sin2x.backward(X_train_sin2x, y_train_sin2x)
            rbf_square2xbox.backward(X_train_square2xbox, y_train_square2xbox)
            rbf_sin2x_noise.backward(sin2x_noise_gen.data[0], sin2x_noise_gen.data[1])
            rbf_square2xbox_noise.backward(square2xbox_noise_gen.data[0], square2xbox_noise_gen.data[1])

            residuals_sin2x.append(np.mean((rbf_sin2x(sin2x_gen.data[0]) - sin2x_gen.data[1]) ** 2))
            residuals_square2xbox.append(np.mean((rbf_square2xbox(square2xbox_gen.data[0]) - square2xbox_gen.data[1]) ** 2))
            residuals_sin2x_noise.append(np.mean((rbf_sin2x_noise(sin2x_noise_gen.data[0]) - sin2x_noise_gen.data[1]) ** 2))
            residuals_square2xbox_noise.append(np.mean((rbf_square2xbox_noise(square2xbox_noise_gen.data[0]) - square2xbox_noise_gen.data[1]) ** 2))

        plt.figure()
        plt.plot(rbf_units_range, residuals_sin2x, label="sin(2x)", marker='o', color='blue')
        plt.plot(rbf_units_range, residuals_square2xbox, label="square(2x)box", marker='o', color='green')
        plt.plot(rbf_units_range, residuals_sin2x_noise, label="sin(2x) with noise", marker='o', color='red')
        plt.plot(rbf_units_range, residuals_square2xbox_noise, label="square(2x)box with noise", marker='o', color='orange')
        
        plt.title('Residual Error vs Number of RBF Units')
        plt.xlabel('Number of RBF Units')
        plt.ylabel('Residual Error (MSE)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_rbf_residuals_vs_units_sigma(self):
        rbf_units_range = range(1, 21)
        sigma_values = [0.1, 0.3, 0.6, 1.0, 2.0]
        
        sin2x_gen = self.data_generator_sin2x
        square2xbox_gen = self.data_generator_square2xbox
        
        sin2x_noise_gen = One_Dim_Function(0, 2 * np.pi, self.samples, function=lambda x: np.sin(2 * x))
        square2xbox_noise_gen = One_Dim_Function(0, 2 * np.pi, self.samples, function=lambda x: np.where(np.sin(x) >= 0, 1, -1))
        sin2x_noise_gen.add_gaussian_noise(0, 0.1)
        square2xbox_noise_gen.add_gaussian_noise(0, 0.1)

        generators = [(sin2x_gen, "sin(2x)"), (square2xbox_gen, "square(2x)box"),
                      (sin2x_noise_gen, "sin(2x) with noise"), (square2xbox_noise_gen, "square(2x)box with noise")]

        for gen, label in generators:
            plt.figure()
            for sigma in sigma_values:
                residual_errors = []

                for rbf_unit in rbf_units_range:
                    rbf_net = RBF(input_dim=1, rbf_dim=rbf_unit)
                    
                    X_train, y_train = gen.data
                    centers = np.linspace(X_train[0], X_train[-1], rbf_unit).reshape(-1, 1)
                    rbf_net.set_C_n_Sigma(centers, np.array([sigma] * rbf_unit))
                    rbf_net.backward(X_train, y_train)

                    residual_errors.append(np.mean((rbf_net(gen.data[0]) - gen.data[1]) ** 2))

                plt.plot(rbf_units_range, residual_errors, label=f"Sigma = {sigma}", marker='o')

            plt.title(f'Residual Error vs Number of RBF Units for {label}')
            plt.xlabel('Number of RBF Units')
            plt.ylabel('Residual Error (MSE)')
            plt.grid(True)
            plt.legend()
            plt.show()

