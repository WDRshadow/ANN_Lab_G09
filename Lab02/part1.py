import unittest

import matplotlib.pyplot as plt
import numpy as np

from utils import RBF, One_Dim_Function, Read_Files, RBF_seq, RBF_Delta_Rule


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
        rbf_unit, sigma = 10, 1.5
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
        self._test(self.data_generator_square2xbox, rbf_net, rbf_unit, sigma, plot_title='2x^2')

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
        
        plt.title('Residual Error vs Number of RBF Units in batch mode')
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

            plt.title(f'Residual Error vs Number of RBF Units for {label} in batch mode')
            plt.xlabel('Number of RBF Units')
            plt.ylabel('Residual Error (MSE)')
            plt.grid(True)
            plt.legend()
            plt.show()


    def test_rbf_residuals_vs_units_seq(self):
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

        s_l, epochs = 0.01, 10
        mode = "b"
        for rbf_unit in rbf_units_range:
            rbf_sin2x = RBF_Delta_Rule(input_dim=1, rbf_dim=rbf_unit, study_rate=s_l, epochs=epochs, mode=mode)
            rbf_square2xbox = RBF_Delta_Rule(input_dim=1, rbf_dim=rbf_unit, study_rate=s_l, epochs=epochs, mode=mode)
            rbf_sin2x_noise = RBF_Delta_Rule(input_dim=1, rbf_dim=rbf_unit, study_rate=s_l, epochs=epochs, mode=mode)
            rbf_square2xbox_noise = RBF_Delta_Rule(input_dim=1, rbf_dim=rbf_unit, study_rate=s_l, epochs=epochs, mode=mode)
            
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
        
        plt.title('Residual Error vs Number of RBF Units in sequential')
        plt.xlabel('Number of RBF Units')
        plt.ylabel('Residual Error (MSE)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_rbf_residuals_vs_units_sigma_seq(self):
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
                    rbf_net = RBF_Delta_Rule(input_dim=1, rbf_dim=rbf_unit)
                    
                    X_train, y_train = gen.data
                    centers = np.linspace(X_train[0], X_train[-1], rbf_unit).reshape(-1, 1)
                    rbf_net.set_C_n_Sigma(centers, np.array([sigma] * rbf_unit))
                    rbf_net.backward(X_train, y_train)

                    residual_errors.append(np.mean((rbf_net(gen.data[0]) - gen.data[1]) ** 2))

                plt.plot(rbf_units_range, residual_errors, label=f"Sigma = {sigma}", marker='o')

            plt.title(f'Residual Error vs Number of RBF Units for {label} in sequential')
            plt.xlabel('Number of RBF Units')
            plt.ylabel('Residual Error (MSE)')
            plt.grid(True)
            plt.legend()
            plt.show()



    def plot_residuals_vs_units_comparison(rbf_units_range, residuals, plot_title):
        plt.figure(figsize=(10, 6))
        
        # Plot lines with different styles for manual and random placements in batch and sequential modes
        plt.plot(rbf_units_range, residuals['manual_batch'], label="Manual - Batch Mode", linestyle='-', marker='o', color='blue')
        plt.plot(rbf_units_range, residuals['manual_sequential'], label="Manual - Sequential Mode", linestyle='--', marker='s', color='green')
        plt.plot(rbf_units_range, residuals['random_batch'], label="Random - Batch Mode", linestyle=':', marker='^', color='red')
        plt.plot(rbf_units_range, residuals['random_sequential'], label="Random - Sequential Mode", linestyle='-.', marker='d', color='orange')
        
        # Titles and labels
        plt.title(f'Residual Error vs Number of RBF Units: {plot_title}')
        plt.xlabel('Number of RBF Units')
        plt.ylabel('Residual Error (MSE)')
        plt.grid(True)
        plt.legend(loc='best')
        
        # Display the plot
        plt.show()

    def plot_rbf_modes_comparison(self, data_generator: One_Dim_Function, function_name: str, rbf_units_range=range(1, 21)):
        residuals_manual_batch = []
        residuals_manual_sequential = []
        residuals_random_batch = []
        residuals_random_sequential = []
        
        for rbf_unit in rbf_units_range:
            rbf_batch = RBF(input_dim=1, rbf_dim=rbf_unit)
            centers_manual = np.linspace(data_generator.data[0][0], data_generator.data[0][-1], rbf_unit).reshape(-1, 1)
            rbf_batch.set_C_n_Sigma(centers_manual, np.array([1.0] * rbf_unit))
            rbf_batch.backward(data_generator.data[0], data_generator.data[1])
            residuals_manual_batch.append(np.mean((rbf_batch(data_generator.data[0]) - data_generator.data[1]) ** 2))
            
            rbf_seq = RBF_Delta_Rule(input_dim=1, rbf_dim=rbf_unit)
            rbf_seq.set_C_n_Sigma(centers_manual, np.array([1.0] * rbf_unit))
            rbf_seq.backward(data_generator.data[0], data_generator.data[1])
            residuals_manual_sequential.append(np.mean((rbf_seq(data_generator.data[0]) - data_generator.data[1]) ** 2))
            
            rbf_random_batch = RBF(input_dim=1, rbf_dim=rbf_unit)
            centers_random = np.random.uniform(data_generator.data[0][0], data_generator.data[0][-1], rbf_unit).reshape(-1, 1)
            rbf_random_batch.set_C_n_Sigma(centers_random, np.array([1.0] * rbf_unit))
            rbf_random_batch.backward(data_generator.data[0], data_generator.data[1])
            residuals_random_batch.append(np.mean((rbf_random_batch(data_generator.data[0]) - data_generator.data[1]) ** 2))
            
            rbf_random_seq = RBF_Delta_Rule(input_dim=1, rbf_dim=rbf_unit)
            rbf_random_seq.set_C_n_Sigma(centers_random, np.array([1.0] * rbf_unit))
            rbf_random_seq.backward(data_generator.data[0], data_generator.data[1])
            residuals_random_sequential.append(np.mean((rbf_random_seq(data_generator.data[0]) - data_generator.data[1]) ** 2))
        
        # residuals = {}
        # residuals['manual_batch'] = residuals_manual_batch
        # residuals['manual_sequential'] = residuals_manual_sequential
        # residuals['random_batch'] = residuals_random_batch
        # residuals['random_sequential'] = residuals_random_sequential
        #
        # self.plot_residuals_vs_units_comparison(rbf_units_range, residuals, function_name)

        plt.figure()
        plt.plot(rbf_units_range, residuals_manual_batch, label="Manual Batch", marker='o', color='blue')
        plt.plot(rbf_units_range, residuals_manual_sequential, label="Manual Sequential", marker='o', color='green')
        plt.plot(rbf_units_range, residuals_random_batch, label="Random Batch", marker='o', color='red')
        plt.plot(rbf_units_range, residuals_random_sequential, label="Random Sequential", marker='o', color='orange')
        
        plt.title(f'Residual Error vs Number of RBF Units - {function_name}')
        plt.xlabel('Number of RBF Units')
        plt.ylabel('Residual Error (MSE)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_run_all_rbf_comparisons(self):
        self.plot_rbf_modes_comparison(self.data_generator_sin2x, 'sin(2x)')
        
        self.plot_rbf_modes_comparison(self.data_generator_square2xbox, 'square(2x)')
        
        sin2x_noise_gen = One_Dim_Function(0, 2 * np.pi, self.samples, function=lambda x: np.sin(2 * x))
        sin2x_noise_gen.add_gaussian_noise(0, 0.5)
        self.plot_rbf_modes_comparison(sin2x_noise_gen, 'sin(2x) with noise')
        
        square2x_noise_gen = One_Dim_Function(0, 2 * np.pi, self.samples, function=lambda x: np.where(np.sin(x) >= 0, 1, -1))
        square2x_noise_gen.add_gaussian_noise(0, 0.5)
        self.plot_rbf_modes_comparison(square2x_noise_gen, 'square(2x)box with noise')
        
