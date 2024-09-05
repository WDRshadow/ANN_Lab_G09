import unittest
from abc import ABC, abstractmethod

from GraphingUtils import plot_line_from_vector
from PointRecognition import *


class LearningAlgorithm(ABC):
    def __init__(self, input_num, output_num, study_rate=0.1, epochs=3000):
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
        self.input_num = input_num
        self.output_num = output_num
        self.epochs = epochs

    def _one_step(self, x: np.ndarray, y: np.ndarray):
        """
        One step of the learning algorithm
        """
        # check if x and y has the same length
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        for i, y_i in enumerate(y):
            y_pred = self(x[i])
            for j, y_j in enumerate(y_i):
                self.w[1:, j] += self.study_rate * (y_j - y_pred[j]) * x[i]

    def train(self, data: (np.ndarray, np.ndarray)):
        """
        The learning loop of the learning algorithm
        :param data: the training data
        """
        x, y = data
        for e in range(self.epochs):
            if self._is_all_points_on_propper_side(data):
                print("all points are correct, n of epochs: ", e)
                return

            self._one_step(x, y)
        print("Values are not linearly separable in specified epochs.")

    def _is_all_points_on_propper_side(self, data: (np.ndarray, np.ndarray)) -> bool:
        """
        Check if all points are on the propper side of the decision line
        :param data: the training data
        :return: True if all points are on the propper side of the decision line, False otherwise
        """
        x, y = data
        y_pred = self(x)
        return np.array_equal(y, y_pred)

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


class PerceptronLearning(LearningAlgorithm):
    """
    Perceptron learning algorithm

    Parameters:
        input_num: the number of input units
        output_num: the number of output units
        threshold: the threshold of the perceptron
        study_rate: the study rate of the perceptron
        epochs: the number of epochs to train the perceptron
    """

    def __init__(self, input_num: int, output_num: int, threshold: float = 0, study_rate: float = 0.1,
                 epochs: int = 3000):
        super().__init__(input_num, output_num, study_rate, epochs)
        self.threshold = threshold

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
        return np.where(np.dot(self.w.transpose(), x.transpose()) > self.threshold, 1, 0).transpose()


class DeltaRuleLearning(LearningAlgorithm):
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
        self.n = 10
        self.input_layer_len = 2
        self.output_layer_len = 1

        self.mA = [-1.0, -0.5]
        self.sigmaA = 0.5
        self.mB = [1.0, 0.5]
        self.sigmaB = 0.5

        self.data = self.generate_data()
        self.randomly_mix_data()

    def generate_data(self):

        self.class1 = self.generate_array(self.n, self.mA, self.sigmaA)
        self.class0 = self.generate_array(self.n, self.mB, self.sigmaB)

        # Determine different ways to stack the classes
        data = np.vstack([self.class1, self.class0])

        labels1 = np.full((self.n, 1), 1)
        labels0 = np.full((self.n, 1), 0)

        labels = np.vstack((labels1, labels0))

        return data, labels

    def randomly_mix_data(self):
        data = np.hstack([self.data[0], self.data[1]])

        # Randomly mix the data points and their labels
        np.random.shuffle(data)

        # Split the data points and their labels
        coords, labels = np.hsplit(data, [2])

        self.data = coords, labels

    @staticmethod
    def generate_array(n, mean, sigma) -> np.ndarray:
        # Convert means to NumPy arrays
        mean = np.array(mean)

        # Generate data for the class
        point_array = np.array([
            np.random.randn(n) * sigma + mean[0],  # Feature 1 for the class
            np.random.randn(n) * sigma + mean[1]  # Feature 2 for the class
        ]).transpose()

        return point_array

    def test_perceptron_learning(self):
        perceptron = PerceptronLearning(self.input_layer_len, self.output_layer_len)
        perceptron.train(self.data)
        plot_line_from_vector(self.class0, self.class1, perceptron.w, title="Perceptron Learning Data")

    def test_delta_rule_learning(self):
        delta_rule = DeltaRuleLearning(self.input_layer_len, self.output_layer_len)
        delta_rule.train(self.data)
        plot_line_from_vector(self.class0, self.class1, delta_rule.w, title="Delta Rule Learning Data")

    def test_all(self):
        self.test_perceptron_learning()
        self.test_delta_rule_learning()
