import numpy as np


class ActivationFunction:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape)


class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(x):
        # for each element in x, if x >= 8, return 1, if x < -8, return -1, else return 2 / (1 + exp(-x)) - 1
        return np.where(x >= 8, 1, np.where(x < -8, -1, 2 / (1 + np.exp(-x)) - 1))

    @staticmethod
    def derivative(x):
        phi = Sigmoid.forward(x)
        return 0.5 * (1 + phi) * (1 - phi)


class ReLU(ActivationFunction):
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)


class Tanh(ActivationFunction):
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x) ** 2
