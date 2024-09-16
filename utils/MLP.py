import numpy as np

from .ActivationFunction import ActivationFunction


class MLP:
    """
    A simple implementation of a Multi-Layer Perceptron (MLP) using numpy.

    Examples
    --------
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> Y = np.array([[0], [1], [1], [0]])
    >>> model = MLP(0.01, 1000)
    >>> model.layers.append(Layer(2, 5))
    >>> model.layers.append(Layer(5, 1))
    >>> model.train(X, Y)
    >>> model.test(X, Y)
    Accuracy: 0.5

    Parameters
    ----------
    study_rate : float
        The learning rate of the model.
    epochs : int
        The number of epochs to train the model.

    """

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
                # (O - Y) * f'(O*)
                delta = layer.backward(dO, X, self.study_rate)
            else:
                # (V^T*delta) * f'(H*)
                delta = layer.backward(dO, Os[index - 1], self.study_rate)
            dO = np.dot(delta, layer.W.T)[:, :-1]

    def train(self, X: np.ndarray, Y: np.ndarray, msg=False):
        losses = []
        for epoch in range(self.epochs):
            O = self.forward(X)
            loss = np.mean((O - Y) ** 2)
            self.backward(X, Y)
            if epoch % 100 == 0 and msg:
                print(f"Epoch {epoch}, Loss: {loss}")
            losses.append(loss)
        return losses

    def test(self, X: np.ndarray, Y: np.ndarray):
        O = self.forward(X)
        accuracy = 1 - np.mean(np.abs(O - Y))
        print(f"Accuracy: {accuracy}")
        return accuracy


class Layer:
    """
    A simple implementation of a layer in a Multi-Layer Perceptron (MLP) using numpy.
    Layer is a linear transformation followed by an activation function.

    Examples
    --------
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> Y = np.array([[0], [1], [1], [0]])
    >>> layer = Layer(2, 1)
    >>> for e in range(1000):
    >>>     layer.backward(layer.forward(X) - Y, X, 0.001)
    >>> accuracy = 1 - np.mean(np.abs(layer.forward(X) - Y))
    0.48365834304155164

    Parameters
    ----------
    input_dim : int
        The number of input features.
    output_dim : int
        The number of output features.
    activation_function : type[ActivationFunction]
        The activation function to use. If None, the default activation function is used. (f(x) = x)
    """

    def __init__(self, input_dim: int, output_dim: int,
                 activation_function: type[ActivationFunction] = ActivationFunction):
        self.W = np.random.randn(input_dim + 1, output_dim) * 0.01
        self.activation_function = activation_function
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
