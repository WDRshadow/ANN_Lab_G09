import unittest

import numpy as np
import matplotlib.pyplot as plt

class PerceptronLearning:
    """
    Perceptron learning algorithm

    Parameters:
        input_num: the number of input units
        output_num: the number of output units
        threshold: the threshold of the perceptron
        study_rate: the study rate of the perceptron
    """

    def __init__(self, input_num: int, output_num: int, threshold: float, study_rate: float):
        self.w = np.zeros((input_num, output_num))
        self.threshold = threshold
        self.study_rate = study_rate
        self.input_num = input_num
        self.output_num = output_num

    def _one_step(self, x: np.ndarray, y: np.ndarray):
        """
        One step of the perceptron learning algorithm
        """
        y_pred = self.forward(x)
        for i, y_unit in enumerate(y_pred):
            if y_unit != y[i]:
                if y[i] == 1:
                    self.w[:, i] += self.study_rate * x
                elif y[i] == 0:
                    self.w[:, i] -= self.study_rate * x
                else:
                    raise ValueError("The value of y should be 0 or 1.")

    def learning_loop(self, data: (np.ndarray, np.ndarray)):
        """
        The learning loop of the perceptron learning algorithm
        :param data: the training data
        """
        x, y = data
        for i in range(len(x)):
            self._one_step(x[i], y[i])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass of the perceptron
        :param x: the input data
        :return: the output data
        """
        return np.where(np.dot(x, self.w) > self.threshold, 1, 0)


class DeltaRuleLearning:
    def __init__(self, input_num, output_num, study_rate):
        """
        Delta rule learning algorithm

        Parameters:
            input_num: the number of input units
            output_num: the number of output units
            study_rate: the study rate of the perceptron
        """
        self.w = np.zeros((input_num, output_num))
        self.study_rate = study_rate
        self.input_num = input_num
        self.output_num = output_num

    def _one_step(self, x: np.ndarray, y: np.ndarray):
        """
        One step of the delta rule learning algorithm
        """
        y_pred = self.forward(x)
        for i, y_unit in enumerate(y_pred):
            e = y[i] - y_unit
            self.w[:, i] += self.study_rate * e * x

    def learning_loop(self, data: (np.ndarray, np.ndarray)):
        """
        The learning loop of the delta rule learning algorithm
        :param data: the training data
        """
        x, y = data
        for i in range(len(x)):
            self._one_step(x[i], y[i])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass of the perceptron
        :param x: the input data
        :return: the output data
        """
        return np.dot(x, self.w)


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.n = 100
        self.threshold = 0
        self.s_r = 0.1
        self.input_layer_len = 10
        self.output_layer_len = 2
        
        mA = [2.0, -1.0]
        sigmaA = 0.8
        mB = [-2.0, 1.0]
        sigmaB = 0.8 

        self.class1 = self.generate_data(self.n, mA, sigmaA)
        self.class0 = self.generate_data(self.n, mB, sigmaB)

    @staticmethod
    def generate_data(n, mean, sigma):
        """
        Generate random data for two classes.

        Parameters:
            n (int): Number of data points per class.
            mean (list or array): Mean for the class (2D).
            sigma (float): Standard deviation for the class.

        Returns:
            point_array (np.ndarray): Data points for the class, shape (2, n).
        """
        # Convert means to NumPy arrays
        mean = np.array(mean)

        # Generate data for class A
        point_array = np.array([
            np.random.randn(n) * sigma + mean[0],  # Feature 1 for class A
            np.random.randn(n) * sigma + mean[1]   # Feature 2 for class A
        ])

        return point_array

    def plot_config(self, ax, title="Generated Data"):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2) 
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)
        ax.legend()

    def plot_data(self, ax):
        """
        Plot the generated data for class 1 and class 0.
        """
        ax.scatter(self.class1[0, :], self.class1[1, :], color='blue', label='Class 1')
        ax.scatter(self.class0[0, :], self.class0[1, :], color='red', label='Class 0')

    def plot_line_from_vector(self, direction_vector: np.ndarray, point=np.array([0, 0]), title="Generated Data"):
        """
        Plot a 2D line given a direction vector and a point.

        Parameters:
            direction_vector (np.ndarray): The direction vector [a, b] of the line.
            point (np.ndarray): A point [x0, y0] through which the line passes (default is the origin).
        """
        fig, ax = plt.subplots()

        ax.quiver(point[0], point[1], direction_vector[0], direction_vector[1], 
               angles='xy', scale_units='xy', scale=1, color='C0')
        try:
            slope = -direction_vector[0] / direction_vector[1]
        except ZeroDivisionError:
            slope = float('inf')

        ax.axline(xy1=point, slope=slope, color='C0', label='Decision Line')

        self.plot_data(ax)
        self.plot_config(ax, title)

        plt.show()

    # def test_perceptron_learning(self):
    #     perceptron = PerceptronLearning(self.input_layer_len, self.output_layer_len, self.threshold, self.s_r)
    #     perceptron.learning_loop(self.data)
    #     # self.plot_line_from_vector(perceptron.w, title="Perceptron Learning Data")

    # def test_delta_rule_learning(self):
    #     delta_rule = DeltaRuleLearning(self.input_layer_len, self.output_layer_len, self.s_r)
    #     delta_rule.learning_loop((self.class0, self.class1))

    def test_test(self):
        direction_vector = [1, 0]
        self.plot_line_from_vector(direction_vector)