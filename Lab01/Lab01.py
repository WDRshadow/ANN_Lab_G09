import unittest

import numpy as np
from GraphingUtils import plot_line_from_vector

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
        # print("y_pred", y_pred, "and y?", y)
        for i, y_unit in enumerate(y_pred):
            # print("y_unit", y_unit, "i:", i)
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
            # print("x[i] and y[i]", x[i], y[i])
            self._one_step(x[i], y[i])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass of the perceptron
        :param x: the input data
        :return: the output data
        """
        # print("checking forward", x, self.w, np.dot(x, self.w))
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
        self.n = 10
        self.threshold = 0
        self.s_r = 0.1
        self.input_layer_len = 10
        self.output_layer_len = 1
        
        self.data = self.generate_data()
        # print(self.data)

    def generate_data(self):
        """
        Generate random data for two classes.
        # TODO: modularize for delta rule

        Parameters:
            No parameters for now
        
        Returns:
            point_array (np.ndarray): Data points for the class, shape (2, n).
        """

        mA = [1.0, -0.5]
        sigmaA = 0.5
        mB = [-1.0, 0]
        sigmaB = 0.5 

        self.class1 = self.generate_array(self.n, mA, sigmaA)
        self.class0 = self.generate_array(self.n, mB, sigmaB)

        # Determine different ways to stack the classes
        data = np.stack([self.class1, self.class0]) 

        labels1 = np.full((self.n, 1), 1)
        labels0 = np.full((self.n, 1), 0)

        labels = np.vstack([labels1, labels0])

        return data, labels

    def generate_array(self, n, mean, sigma):
        """
        Generate random data for a class.

        Parameters:
            n (int): Number of data points per class.
            mean (list or array): Mean for the class (2D).
            sigma (float): Standard deviation for the class.

        Returns:
            point_array (np.ndarray): Data points for the class, shape (2, n).
        """
        # Convert means to NumPy arrays
        mean = np.array(mean)

        # Generate data for the class
        point_array = np.array([
            np.random.randn(n) * sigma + mean[0],   # Feature 1 for the class
            np.random.randn(n) * sigma + mean[1]    # Feature 2 for the class
        ])

        return point_array

    # def test_perceptron_learning(self):
    #     perceptron = PerceptronLearning(self.input_layer_len, self.output_layer_len, self.threshold, self.s_r)
    #     perceptron.learning_loop(self.data)
        # self.plot_line_from_vector(perceptron.w, title="Perceptron Learning Data")

    # def test_delta_rule_learning(self):
    #     delta_rule = DeltaRuleLearning(self.input_layer_len, self.output_layer_len, self.s_r)
    #     delta_rule.learning_loop(self.data)
    #     self.plot_line_from_vector(delta_rule.w, title="Delta Rule Learning Data")

    def test_test(self):
        direction_vector = [1, 0]
        plot_line_from_vector(self.class0, self.class1, direction_vector)