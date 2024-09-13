import numpy as np
from matplotlib import pyplot as plt


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

    def plot(self):
        """
        Plot the generated data
        """
        fig, ax = plt.subplots()

        labels = self.data[1].flatten()
        class1 = self.data[0][labels == 1]
        class0 = self.data[0][labels == -1]

        ax.scatter(class1.transpose()[0, :], class1.transpose()[1, :], color='blue', label='Class 1')
        ax.scatter(class0.transpose()[0, :], class0.transpose()[1, :], color='red', label='Class 0')

        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title("Generated Data")
        ax.legend()

        plt.show()


class DataGenerator2(DataGenerator):
    def __init__(self, n=100, mA=None, sigmaA = 0.5, mB=None, sigmaB = 0.5, mA2=None, sigmaA2 = 0.5):
        if mA2 is None:
            mA2 = [-1.0, 0.0]
        self.mA2 = mA2
        self.sigmaA2 = sigmaA2
        super().__init__(n, mA, sigmaA, mB, sigmaB)

        self.data = self._generate_data()
        self._randomly_mix_data()

    def _generate_data(self):

        class1 = self._generate_array(self.n, self.mA, self.sigmaA)
        class0 = self._generate_array(self.n, self.mB, self.sigmaB)
        class1_2 = self._generate_array(self.n, self.mA2, self.sigmaA2)

        # Determine different ways to stack the classes
        data = np.vstack([class1, class0, class1_2])

        labels1 = np.full((self.n, 1), 1)
        labels0 = np.full((self.n, 1), -1)
        labels1_2 = np.full((self.n, 1), 1)

        labels = np.vstack((labels1, labels0, labels1_2))

        return data, labels


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
            x[i] = x[i - 1] + (self.beta * x[i - 1 - self.tau]) / (1 + x[i - 1 - self.tau] ** 10) - self.gamma * x[i - 1]
        return x

    def generate_data(self):
        X = self.generate_mackey_glass()
        t = np.arange(301, 1500 + 1)
        inputs = [X[i - 20: i: 5] for i in t]
        outputs = [X[i + 5] for i in t]
        return np.array(inputs), np.array(outputs)