import unittest

import numpy as np
from GraphingUtils import plot_line_from_vector, plot_line_from_weights
from PointRecognition import *


class PerceptronLearning:
    """
    Perceptron learning algorithm

    Parameters:
        input_num: the number of input units
        output_num: the number of output units
        threshold: the threshold of the perceptron
        study_rate: the study rate of the perceptron
        epochs: the number of epochs to train the perceptron
    """

    def __init__(self, input_num: int, output_num: int, threshold: float, study_rate: float = 0.1, epochs: int = 100):
        self.w = np.ones((input_num + 1, output_num))
        self.threshold = threshold
        self.study_rate = study_rate
        self.input_num = input_num
        self.output_num = output_num
        self.epochs = epochs

    def _one_step(self, x: np.ndarray, y: np.ndarray):
        """
        One step of the perceptron learning algorithm
        """
        y_pred = self.forward(x)
        for i, y_i in enumerate(y_pred):
            for j, y_j in enumerate(y_i):
                if (all_points_on_propper_side(self.w, x, y)): 
                    return
            
                if (not is_point_on_propper_side(self.w, x[i], y[i, j])):
                    self.w[:, j] += self.study_rate * (y[i, j] - y_j) * np.insert(x[i], 0, 1)
        
    def learning_loop(self, data: (np.ndarray, np.ndarray)):
        """
        The learning loop of the perceptron learning algorithm
        :param data: the training data
        """
        x, y = data
        for e in range(self.epochs):
            if (all_points_on_propper_side(self.w, x, y)): 
                print("all points are correct, n of epochs: ", e)
                return

            self._one_step(x, y)
        raise RuntimeError("Values are not linearly separable in specified epochs.") 

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass of the perceptron
        :param x: the input data
        :return: the output data
        """
        # Add bias term to input
        x = np.insert(x, 2, 0, axis=1)
        # print("checking forward", x, self.w, np.dot(x, self.w))
        return np.where(np.dot(self.w.transpose(), x.transpose()) > self.threshold, 1, 0).transpose()


class DeltaRuleLearning:
    def __init__(self, input_num, output_num, study_rate=0.1, epochs=100):
        """
        Delta rule learning algorithm

        Parameters:
            input_num: the number of input units
            output_num: the number of output units
            study_rate: the study rate of the perceptron
            epochs: the number of epochs to train the perceptron
        """
        self.w = np.zeros((input_num + 1, output_num))
        self.study_rate = study_rate
        self.input_num = input_num
        self.output_num = output_num
        self.epochs = epochs

    def _one_step(self, x: np.ndarray, y: np.ndarray):
        """
        One step of the delta rule learning algorithm
        """
        y_pred = self.forward(x)
        for i, y_i in enumerate(y_pred):
            for j, y_j in enumerate(y_i):
                e = y[i, j] - y_j
                self.w[:, j] += self.study_rate * e * np.insert(x[i], 0, 1)
        # TODO: check if the classification is correct

    def learning_loop(self, data: (np.ndarray, np.ndarray)):
        """
        The learning loop of the delta rule learning algorithm
        :param data: the training data
        """
        x, y = data
        for e in range(self.epochs):
            self._one_step(x, y)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass of the perceptron
        :param x: the input data
        :return: the output data
        """
        # Add bias term to input
        x = np.insert(x, 2, 0, axis=1)
        return np.dot(self.w.transpose(), x.transpose()).transpose()


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.n = 10
        self.threshold = 0
        self.s_r = 0.1
        self.input_layer_len = 2
        self.output_layer_len = 1

        self.data = self.generate_data()
        # print(self.data)

    def generate_data(self):
        """
        Generate random data for two classes.
        # TODO: modularize for delta rule
        
        Returns:
            data (np.ndarray): Data points for the classes, shape (2*n, 2). Example: [[1, 2], [3, 4], ...]
            labels (np.ndarray): Labels for the classes, shape (2*n, 1). Example: [[1], [0], ...]
        """

        mA = [-1.0, -0.5]
        sigmaA = 0.5
        mB = [1.0, 0.5]
        sigmaB = 0.5

        self.class1 = self.generate_array(self.n, mA, sigmaA)
        self.class0 = self.generate_array(self.n, mB, sigmaB)

        # Determine different ways to stack the classes
        data = np.vstack([self.class1, self.class0])

        labels1 = np.full((self.n, 1), 1)
        labels0 = np.full((self.n, 1), 0)

        labels = np.vstack((labels1, labels0))

        return data, labels

    @staticmethod
    def generate_array(n, mean, sigma) -> np.ndarray:
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
            np.random.randn(n) * sigma + mean[0],  # Feature 1 for the class
            np.random.randn(n) * sigma + mean[1]  # Feature 2 for the class
        ]).transpose()

        return point_array

    def test_perceptron_learning(self):
        perceptron = PerceptronLearning(self.input_layer_len, self.output_layer_len, self.threshold, self.s_r)
        perceptron.learning_loop(self.data)
        plot_line_from_vector(self.class0, self.class1, perceptron.w, title="Perceptron Learning Data")

    def test_delta_rule_learning(self):
        delta_rule = DeltaRuleLearning(self.input_layer_len, self.output_layer_len, self.s_r)
        delta_rule.learning_loop(self.data)
        plot_line_from_vector(self.class0, self.class1, delta_rule.w, title="Delta Rule Learning Data")

    # def test_test(self):
    #     direction_vector = [0, 1, 0]
    #     plot_line_from_vector(self.class0, self.class1, direction_vector)

    #     point = (1, 1)
    #     print(is_point_on_positive_side(direction_vector, point))  # Output: True (depends on dir vector)
    #     print(is_point_on_propper_side(direction_vector, point, 1))  # Output: True (depends on dir vector)

    #     point = (-1, -1)
    #     print(is_point_on_positive_side(direction_vector, point))  # Output: False (depends on dir vector)
    #     print(is_point_on_propper_side(direction_vector, point, -1))  # Output: True (depends on dir vector)

    #     points, values = self.data
    #     print(all_points_on_propper_side(direction_vector, points, values)) # Output: Depends on the point generation