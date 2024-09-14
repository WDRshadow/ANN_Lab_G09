from abc import ABC, abstractmethod

import numpy as np


class SingleLevelPerceptron(ABC):
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
        for _ in range(self.epochs):
            self._one_step(x, y, learning_type)
        if self._is_all_points_on_propper_side(data):
            return
        print("Values are not linearly separable in specified epochs.")

    @abstractmethod
    def _is_all_points_on_propper_side(self, data: (np.ndarray, np.ndarray)) -> bool:
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


class PerceptronLearning(SingleLevelPerceptron):
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


class DeltaRuleLearning(SingleLevelPerceptron):
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