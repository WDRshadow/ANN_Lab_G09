import time
import unittest
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from sympy import false

from DataGenerator import DataGenerator


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

        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
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


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.study_rate = 0.001
        self.epochs = 3000

        self.data_generator = DataGenerator(
            n=100,
            mA=[0.0, 2.0],
            sigmaA = 0.3,
            mB=[-0.0, -0.0],
            sigmaB = 0.3
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

    def test_all(self, plot=True):
        perceptron = PerceptronLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        perceptron.train(self.data_generator.data)
        delta_rule = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule.train(self.data_generator.data, learning_type="b")
        if plot:
            self.two_in_one_plot(perceptron, delta_rule)
            self.error_two_in_one_plot(perceptron, delta_rule)
        return perceptron, delta_rule

    def test_left_right_side(self):
        self.data_generator = DataGenerator(
            n=10,
            mA=[-1.0, 0.0],
            sigmaA = 0.5,
            mB=[1.0, 0.0],
            sigmaB = 0.5
        )
        self.test_all()

    def test_different_study_rate(self):
        perceptron, delta_rule = self.test_all(plot=false)
        self.study_rate = 0.1
        perceptron1, delta_rule1 = self.test_all(plot=false)
        self.two_in_one_plot(perceptron, perceptron1)
        self.error_two_in_one_plot(perceptron, perceptron1)
        self.two_in_one_plot(delta_rule, delta_rule1)
        self.error_two_in_one_plot(delta_rule, delta_rule1)


    def test_learning_type(self):
        # calculate systemds time
        time_0 = time.time()
        delta_rule = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule.train(self.data_generator.data, learning_type="s")
        print("Time for sequential: ", time.time() - time_0)
        time_0 = time.time()
        delta_rule2 = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule2.train(self.data_generator.data, learning_type="b")
        print("Time for batch: ", time.time() - time_0)
        self.two_in_one_plot(delta_rule, delta_rule2)
        self.error_two_in_one_plot(delta_rule, delta_rule2)

    def two_in_one_plot(self, model1: LearningAlgorithm, model2: LearningAlgorithm):
        fig, ax = plt.subplots()

        y_0 = -model1.w[0] / model1.w[2]
        point = np.array([0, y_0[0]])

        # plot the normal vector to the decision line at the point
        ax.quiver(*point, model1.w[1], -model1.w[2], color='C0')

        try:
            slope = -model1.w[1] / model1.w[2]
            slope = slope[0]
        except ZeroDivisionError:
            slope = float('inf')

        ax.axline(xy1=point, slope=slope, color='C0', label='Decision Line' + model1.__class__.__name__)

        y_0 = -model2.w[0] / model2.w[2]
        point = np.array([0, y_0[0]])

        # plot the normal vector to the decision line at the point
        ax.quiver(*point, model2.w[1], -model2.w[2], color='C1')

        try:
            slope = -model2.w[1] / model2.w[2]
            slope = slope[0]
        except ZeroDivisionError:
            slope = float('inf')

        ax.axline(xy1=point, slope=slope, color='C1', label='Decision Line' + model2.__class__.__name__)

        labels = self.data_generator.data[1].flatten()
        class1 = self.data_generator.data[0][labels == 1]
        class0 = self.data_generator.data[0][labels == -1]

        ax.scatter(class1.transpose()[0, :], class1.transpose()[1, :], color='blue', label='Class 1')
        ax.scatter(class0.transpose()[0, :], class0.transpose()[1, :], color='red', label='Class 0')

        ax.set_xlim(-2, 5)
        ax.set_ylim(-2, 5)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title("Generated Data")
        ax.legend()

        plt.show()

    @staticmethod
    def error_two_in_one_plot(model1: LearningAlgorithm, model2: LearningAlgorithm):
        plt.plot(model1.errors, color='C0', label=model1.__class__.__name__)
        plt.plot(model2.errors, color='C1', label=model2.__class__.__name__)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.title("Error over epochs")
        plt.legend()
        plt.show()
