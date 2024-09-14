import unittest

import numpy as np
from matplotlib import pyplot as plt


class DataGenerator:
    """
    Data generator for the perceptron learning algorithm

    Properties:
        data (np.ndarray): the data points and their labels
    """

    def __init__(self, n=100, mA=None, sigmaA=0.2, mB=None, sigmaB=0.3):
        self.n = n

        if mA is None:
            mA = [1.0, 0.5]
        if mB is None:
            mB = [-0.0, -0.1]
        self.mA = mA
        self.sigmaA = sigmaA
        self.mB = mB
        self.sigmaB = sigmaB

        self.data = self._generate_data()
        self.data_copy = (self.data[0].copy(), self.data[1].copy())
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

        classA = self._generate_array(self.n, self.mA, self.sigmaA)
        classB = self._generate_array(self.n, self.mB, self.sigmaB)

        # Determine different ways to stack the classes
        data = np.vstack([classA, classB])

        labelsA = np.full((self.n, 1), 1)
        labelsB = np.full((self.n, 1), -1)

        labels = np.vstack((labelsA, labelsB))

        return data, labels

    def _randomly_mix_data(self):
        data = np.hstack([self.data[0], self.data[1]])

        # Randomly mix the data points and their labels
        np.random.shuffle(data)

        # Split the data points and their labels
        coords, labels = np.hsplit(data, [2])

        self.data = coords, labels

    def _get_class_data(self):
        data = np.hstack([self.data_copy[0], self.data_copy[1]])
        data_A = data[:self.n]
        data_B = data[self.n:]
        return data_A, data_B

    def randomly_remove_data(self, A_perc, B_perc):
        """
        Randomly remove a percentage of the data points from class A and B and return the removed data points

        :param A_perc: The percentage of data points to remove from class A
        :param B_perc: The percentage of data points to remove from class B
        :return: The removed data points
        """
        data_A = self._get_class_data()[0]
        data_B = self._get_class_data()[1]
        # Randomly remove data points from class A
        np.random.shuffle(data_A)
        n_A = int(len(data_A) * (1 - A_perc))
        data_A_new = data_A[:n_A]
        # Randomly remove data points from class B
        np.random.shuffle(data_B)
        n_B = int(len(data_B) * (1 - B_perc))
        data_B_new = data_B[:n_B]
        # Combine the data points
        data = np.vstack([data_A_new, data_B_new])
        # separate the data points and their labels
        coords, labels = np.hsplit(data, [2])
        self.data = coords, labels
        self._randomly_mix_data()
        # collect the removed data points
        removed_data_A = data_A[n_A:]
        removed_data_B = data_B[n_B:]
        removed_data = np.vstack([removed_data_A, removed_data_B])
        return np.hsplit(removed_data, [2])

    def randomly_remove_specific_data(self, low_perc=0.2, high_perc=0.8):
        """
        Randomly remove 20% of the data points from class A where x0 < 0 and 80% of the data points from class A where x0 >= 0.
        Return the removed data points

        :return: The removed data points
        """
        data_A = self._get_class_data()[0]
        # select the points from class A where x0 < 0 and randomly remove 20% of them
        data_A_low = data_A[data_A[:, 0] < 0]
        np.random.shuffle(data_A_low)
        n_A = int(len(data_A_low) * (1 - low_perc))
        data_A_low_new = data_A_low[:n_A]
        # select the points from class A where x0 >= 0 and randomly remove 80% of them
        data_A_high = data_A[data_A[:, 0] >= 0]
        np.random.shuffle(data_A_high)
        n_A = int(len(data_A_high) * (1 - high_perc))
        data_A_high_new = data_A_high[:n_A]
        # Combine the data points
        data = np.vstack([data_A_low_new, data_A_high_new, self._get_class_data()[1]])
        # separate the data points and their labels
        coords, labels = np.hsplit(data, [2])
        self.data = coords, labels
        self._randomly_mix_data()
        # collect the removed data points
        removed_data_A = np.vstack([data_A_low[n_A:], data_A_high[n_A:]])
        return np.hsplit(removed_data_A, [2])

    def reset_data(self):
        """
        Reset the data to the original data
        """
        self.data = self.data_copy
        self._randomly_mix_data()

    def plot(self):
        """
        Plot the generated data
        """
        fig, ax = plt.subplots()

        labels = self.data[1].flatten()
        classA = self.data[0][labels == 1]
        classB = self.data[0][labels == -1]

        ax.scatter(classA.transpose()[0, :], classA.transpose()[1, :], color='red', label='Class A')
        ax.scatter(classB.transpose()[0, :], classB.transpose()[1, :], color='blue', label='Class B')

        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title("Generated Data")
        ax.legend()

        plt.show()


class DataGenerator2(DataGenerator):
    def __init__(self, n=100, mA=None, sigmaA=0.2, mB=None, sigmaB=0.3, mA2=None, sigmaA2=0.2):
        if mA2 is None:
            mA2 = [-1.5, 0.3]
        self.mA2 = mA2
        self.sigmaA2 = sigmaA2
        super().__init__(n, mA, sigmaA, mB, sigmaB)

    def _generate_data(self):
        classA = self._generate_array(self.n, self.mA, self.sigmaA)
        classA_2 = self._generate_array(self.n, self.mA2, self.sigmaA2)
        classB = self._generate_array(self.n * 2, self.mB, self.sigmaB)

        # Determine different ways to stack the classes
        data = np.vstack([classA, classA_2, classB])

        labelsA = np.full((self.n, 1), 1)
        labelsA_2 = np.full((self.n, 1), 1)
        labelsB = np.full((self.n * 2, 1), -1)

        labels = np.vstack((labelsA, labelsA_2, labelsB))

        return data, labels

    def _get_class_data(self):
        data = np.hstack([self.data_copy[0], self.data_copy[1]])
        data_A = data[:(self.n * 2)]
        data_B = data[(self.n * 2):]
        return data_A, data_B


class MackeyGlass:
    def __init__(self, n=1506, beta=0.2, gamma=0.1, tau=25):
        self.n = n
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def generate_mackey_glass(self):
        x = np.zeros(self.n)
        x[0] = 1.5
        for i in range(1, self.n):
            x[i] = x[i - 1] + (self.beta * x[i - 1 - self.tau]) / (1 + x[i - 1 - self.tau] ** 10) - self.gamma * x[
                i - 1]
        return x

    def generate_data(self, f=301, t=1500, s=5, d=20):
        X = self.generate_mackey_glass()
        t = np.arange(f, t + 1)
        inputs = [X[i - d: i: s] for i in t]
        outputs = [X[i + s] for i in t]
        return np.array(inputs), np.array(outputs)


class Test(unittest.TestCase):
    def test_1(self):
        data_generator = DataGenerator()
        data_generator.plot()

    def test_2(self):
        data_generator = DataGenerator2()
        data_generator.plot()

    def test_3(self):
        mackey_glass = MackeyGlass()
        inputs, outputs = mackey_glass.generate_data()
        self.assertEqual(inputs.shape, (1200, 4))
        self.assertEqual(outputs.shape, (1200,))

    def test_1_25_each(self):
        data_generator = DataGenerator()
        data_generator.plot()
        points = data_generator.randomly_remove_data(0.25, 0.25)
        print(points)
        data_generator.plot()

    def test_1_50_A(self):
        data_generator = DataGenerator()
        data_generator.plot()
        points = data_generator.randomly_remove_data(0.5, 0)
        print(points)
        data_generator.plot()

    def test_2_specific(self):
        data_generator = DataGenerator2()
        data_generator.plot()
        points = data_generator.randomly_remove_specific_data()
        print(points)
        data_generator.plot()
