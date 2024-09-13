import numpy as np


class Module:
    def __init__(self, study_rate: float, epochs: int):
        self.study_rate = study_rate
        self.epochs = epochs
        self.layers = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = X
        for layer in self.layers:
            Y = layer.forward(Y)
        return Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def backward(self, X: np.ndarray, Y: np.ndarray):
        I = X
        Os = []
        for layer in self.layers:
            I = layer.forward(I)
            Os.append(I)
        dO = Os[-1] - Y
        for i, layer in enumerate(reversed(self.layers)):
            index = len(self.layers) - i - 1
            if index == 0:
                delta = layer.backward(dO, X, self.study_rate)
            else:
                delta = layer.backward(dO, Os[index - 1], self.study_rate)
            dO = np.dot(layer.W, delta.T)[1:].T

    def train(self, X: np.ndarray, Y: np.ndarray, msg=False):
        for epoch in range(self.epochs):
            O = self.forward(X)
            loss = np.mean((O - Y) ** 2)
            self.backward(X, Y)
            if epoch % 100 == 0 and msg:
                print(f"Epoch {epoch}, Loss: {loss}")

    def test(self, X: np.ndarray, Y: np.ndarray):
        O = self.forward(X)
        accuracy = 1 - np.mean(np.abs(O - Y))
        print(f"Accuracy: {accuracy}")


class ActivationFunction:
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        pass


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


class Layer:
    def __init__(self, input_dim: int, output_dim: int, activation_function: type[ActivationFunction], study_rate=0.001):
        self.W = np.random.randn(input_dim + 1, output_dim) * 0.01
        self.activation_function = activation_function
        self.study_rate = study_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        X = np.insert(inputs, inputs.shape[1], 1, axis=1)
        O_ = np.dot(X, self.W)
        O = self.activation_function.forward(O_)
        return O

    def backward(self, dO: np.ndarray, X: np.ndarray, learning_rate: float) -> np.ndarray:
        # delta = dO * f'(WX)
        X = np.insert(X, X.shape[1], 1, axis=1)
        O_ = np.dot(X, self.W)
        delta = dO * self.activation_function.derivative(O_)
        self.W -= learning_rate * np.dot(X.T, delta)
        return delta
