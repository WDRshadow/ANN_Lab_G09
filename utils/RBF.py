import numpy as np

from .MLP import Layer, MLP


class RBF(MLP):
    def __init__(self, input_dim: int, rbf_dim: int, study_rate=0.001, epochs=1000):
        super().__init__(study_rate, epochs)
        self.layers.append(RBFLayer(input_dim, rbf_dim))
        self.layers.append(Layer(rbf_dim, 1))

    def backward(self, X: np.ndarray, Y: np.ndarray):
        X = self.layers[0].forward(X)
        X = np.insert(X, X.shape[1], 1, axis=1)
        self.layers[1].W = np.linalg.pinv(X) @ Y

    def train(self, X: np.ndarray, Y: np.ndarray, msg=False):
        for _ in range(self.epochs):
            self.layers[0].backward(None, X, self.study_rate)
        self.backward(X, Y)

    def set_C_n_Sigma(self, C: np.ndarray, Sigma: np.ndarray):
        self.layers[0].C = C
        self.layers[0].Sigma = Sigma


class RBFLayer(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.C = np.random.rand(output_dim, input_dim) * 2 - 1
        self.Sigma = np.array([1.0] * output_dim)

    @staticmethod
    def _rbf(x: np.ndarray, center: np.ndarray, sigma: float):
        """Kernel function"""
        return np.exp(-np.linalg.norm(x - center, axis=1) ** 2 / (2 * sigma ** 2))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        X = np.array([inputs] * self.C.shape[0])
        return np.array([self._rbf(x, c, s) for x, c, s in zip(X, self.C, self.Sigma)]).T

    def backward(self, dO: np.ndarray, X: np.ndarray, learning_rate: float) -> np.ndarray:
        for x in X:
            x = np.array([x] * self.C.shape[0])
            distances = np.linalg.norm(x - self.C, axis=1)
            winner_idx = np.argmin(distances)
            self.C[winner_idx] += learning_rate * (x[0] - self.C[winner_idx])
        return dO
