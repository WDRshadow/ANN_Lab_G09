import unittest

import numpy as np


def generate_data(input_num, output_num, num) -> (np.ndarray, np.ndarray):
    """
    Generate random data for the perceptron learning algorithm
    """
    return np.random.rand(num, input_num), np.random.randint(0, 2, (num, output_num))


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
    def test_both_learning(self):
        n = 100
        threshold = 0
        s_r = 0.1
        input_layer_len = 10
        output_layer_len = 2
        data = generate_data(10, 2, n)
        perceptron = PerceptronLearning(input_layer_len, output_layer_len, threshold, s_r)
        perceptron.learning_loop(data)
        delta_rule = DeltaRuleLearning(input_layer_len, output_layer_len, s_r)
        delta_rule.learning_loop(data)
