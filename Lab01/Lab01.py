import time
import unittest
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class LearningAlgorithm(ABC):
    def __init__(self, input_num, output_num, study_rate=0.01, epochs=1000, threshold=0.0):
        """
        Basic learning algorithm

        Parameters:
            input_num: the number of input units
            output_num: the number of output units
            study_rate: the study rate of the perceptron
            epochs: the number of epochs to train the perceptron
        """
        self.w = np.zeros((input_num + 1, output_num))
        self.study_rate = study_rate
        self.threshold = threshold
        self.input_num = input_num
        self.output_num = output_num
        self.epochs = epochs

        self.errors = []

    def _one_step(self, x: np.ndarray, y: np.ndarray, learning_type="s"):
        """
        One step of the learning algorithm
        """
        if learning_type == "s":
            for i, y_i in enumerate(y):
                y_pred = self(x[i])
                for j, y_j in enumerate(y_i):
                    x_bias = np.insert(x[i], 0, 1)
                    self.w[:, j] += self.study_rate * (y_j - y_pred[j]) * x_bias
        elif learning_type == "b":
            y_pred = self(x)
            for j in range(self.output_num):
                x_bias = np.insert(x, 0, 1, axis=1)
                self.w[:, j] += self.study_rate * np.sum((y[:, j] - y_pred[:, j]) * x_bias.transpose(), axis=1)
        self.errors.append(np.sum((y - self(x)) ** 2) / 2)

    def train(self, data: (np.ndarray, np.ndarray), learning_type="s"):
        """
        The learning loop of the learning algorithm
        :param learning_type: the type of the learning algorithm (s or b)
        :param data: the training data
        """
        x, y = data
        # check if x and y has the same length
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        for e in range(self.epochs):
            if self._is_all_points_on_propper_side(data):
                print("all points are correct, n of epochs: ", e)
                return
            self._one_step(x, y, learning_type)
        print("Values are not linearly separable in specified epochs.")

    @abstractmethod
    def _is_all_points_on_propper_side(self, data: (np.ndarray, np.ndarray)) -> bool:
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    def plot(self, data: (np.ndarray, np.ndarray), title="Generated Data"):
        """
        Plot a 2D line given a direction vector and a point.

        The line should be 0 = bias + a*x1 + b*x2.

        Parameters:
            data (np.ndarray): The data to plot.
            title (str): Title of the plot.
        """
        fig, ax = plt.subplots()

        y_0 = -self.w[0] / self.w[2]
        point = np.array([0, y_0[0]])

        # plot the normal vector to the decision line at the point
        ax.quiver(*point, self.w[1], -self.w[2], color='C0', label='Normal Vector')

        try:
            slope = -self.w[1] / self.w[2]
            slope = slope[0]
        except ZeroDivisionError:
            slope = float('inf')

        ax.axline(xy1=point, slope=slope, color='C0', label='Decision Line')

        labels = data[1].flatten()
        class1 = data[0][labels == 1]
        class0 = data[0][labels == -1]

        ax.scatter(class1.transpose()[0, :], class1.transpose()[1, :], color='blue', label='Class 1')
        ax.scatter(class0.transpose()[0, :], class0.transpose()[1, :], color='red', label='Class 0')

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)
        ax.legend()

        plt.show()

    def plot_errors(self):
        """
        Plot the errors over the epochs
        """
        plt.plot(self.errors)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Error over epochs")
        plt.show()


class PerceptronLearning(LearningAlgorithm):
    def _is_all_points_on_propper_side(self, data: (np.ndarray, np.ndarray)) -> bool:
        """
        Check if all points are on the propper side of the decision line
        :param data: the training data
        :return: True if all points are on the propper side of the decision line, False otherwise
        """
        x, y = data
        y_pred = self(x)
        return np.array_equal(y, y_pred)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass of the perceptron
        :param x: the input data
        :return: the output data
        """
        # check if x is a 2 dimensional array
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        # Add bias term to input
        x = np.insert(x, 0, 1, axis=1)
        return np.where(np.dot(self.w.transpose(), x.transpose()) > self.threshold, 1, -1).transpose()


class DeltaRuleLearning(LearningAlgorithm):
    def _is_all_points_on_propper_side(self, data: (np.ndarray, np.ndarray)) -> bool:
        """
        Check if all points are on the propper side of the decision line
        :param data: the training data
        :return: True if all points are on the propper side of the decision line, False otherwise
        """
        x, y = data
        if self.w[2] == 0:
            return False
        # x2 = (-bias - a*x1) / b
        x2 = (-self.w[0] - self.w[1] * x.transpose()[0]) / self.w[2]
        y_pred = np.where(x.transpose()[1] < x2, -1, 1).reshape(-1, 1)
        return np.array_equal(y, y_pred)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass of the perceptron
        :param x: the input data
        :return: the output data
        """
        # check if x is a 2 dimensional array
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        # Add bias term to input
        x = np.insert(x, 0, 1, axis=1)
        return np.dot(self.w.transpose(), x.transpose()).transpose()


class DataGenerator:
    """
    Data generator for the perceptron learning algorithm

    Properties:
        data (np.ndarray): the data points and their labels
    """
    def __init__(self, n=100, mA=None, sigmaA = 0.5, mB=None, sigmaB = 0.5):
        self.n = n

        if mA is None:
            mA = [1.0, 0.5]
        if mB is None:
            mB = [-1.0, -0.5]
        self.mA = mA
        self.sigmaA = sigmaA
        self.mB = mB
        self.sigmaB = sigmaB

        self.data = self._generate_data()
        self._randomly_mix_data()


    @staticmethod
    def _generate_array(n, mean, sigma) -> np.ndarray:
        # Convert means to NumPy arrays
        mean = np.array(mean)

        # Generate data for the class
        point_array = np.array([
            np.random.randn(n) * sigma + mean[0],  # Feature 1 for the class
            np.random.randn(n) * sigma + mean[1]  # Feature 2 for the class
        ]).transpose()

        return point_array

    def _generate_data(self):

        class1 = self._generate_array(self.n, self.mA, self.sigmaA)
        class0 = self._generate_array(self.n, self.mB, self.sigmaB)

        # Determine different ways to stack the classes
        data = np.vstack([class1, class0])

        labels1 = np.full((self.n, 1), 1)
        labels0 = np.full((self.n, 1), -1)

        labels = np.vstack((labels1, labels0))

        return data, labels

    def _randomly_mix_data(self):
        data = np.hstack([self.data[0], self.data[1]])

        # Randomly mix the data points and their labels
        np.random.shuffle(data)

        # Split the data points and their labels
        coords, labels = np.hsplit(data, [2])

        self.data = coords, labels



class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.study_rate = 0.001
        self.epochs = 3000

        self.data_generator = DataGenerator(
            n=100,
            mA=[0.0, 2.0],
            sigmaA = 0.5,
            mB=[-0.0, -0.0],
            sigmaB = 0.5
        )

    def test_perceptron_learning(self):
        perceptron = PerceptronLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        perceptron.train(self.data_generator.data)
        perceptron.plot(self.data_generator.data, title="Perceptron Learning Data")
        perceptron.plot_errors()

    def test_delta_rule_learning(self):
        delta_rule = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule.train(self.data_generator.data, learning_type="b")
        delta_rule.plot(self.data_generator.data, title="Delta Rule Learning Data")
        delta_rule.plot_errors()

    def test_all(self):
        self.test_perceptron_learning()
        self.test_delta_rule_learning()

    def test_left_right_side(self):
        self.data_generator = DataGenerator(
            n=100,
            mA=[-1.0, 0.0],
            sigmaA = 0.5,
            mB=[1.0, 0.0],
            sigmaB = 0.5
        )
        self.test_perceptron_learning()
        self.test_delta_rule_learning()

    def test_different_study_rate(self):
        self.test_perceptron_learning()
        self.test_delta_rule_learning()
        self.study_rate = 0.1
        self.test_perceptron_learning()
        self.test_delta_rule_learning()


    def test_learning_type(self):
        # calculate systemds time
        time_0 = time.time()
        delta_rule = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule.train(self.data_generator.data, learning_type="s")
        delta_rule.plot(self.data_generator.data, title="Delta Rule Learning Data with learning type sequential")
        delta_rule.plot_errors()
        print("Time for sequential: ", time.time() - time_0)
        time_0 = time.time()
        delta_rule = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule.train(self.data_generator.data, learning_type="b")
        delta_rule.plot(self.data_generator.data, title="Delta Rule Learning Data with learning type batch")
        delta_rule.plot_errors()
        print("Time for batch: ", time.time() - time_0)

