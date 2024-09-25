import numpy as np


class SOM_2D:
    def __init__(self, m, n, dim, learning_rate=0.5, sigma_0=1.0):
        """
        Parameters:
            m: Size of the SOM, columns
            n: Size of the SOM, rows
            dim: dimension of the input data
            learning_rate: learning rate of the SOM
        """
        self.m = m
        self.n = n
        self.dim = dim
        self.learning_rate = learning_rate
        self.sigma_0 = sigma_0
        self.weights = np.random.rand(m, n, dim)

    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), (self.m, self.n))
        return bmu_index

    def _sigma(self, iteration, max_iter):
        """Sigma function, which decreases with the iteration number"""
        return self.sigma_0 * np.exp(- iteration ** 2 / max_iter)

    def _distance_to_bmu(self, bmu_index, i, j):
        """Calculate the distance between the current node and the BMU"""
        return np.linalg.norm(np.array([i, j]) - np.array(bmu_index))

    def _neighbourhood(self, distance_to_bmu, iteration, max_iter):
        """ Neighbourhood function of the node on the BMU, using a Gaussian function """
        neighbourhood = np.exp(
            -distance_to_bmu ** 2 / (2 * self._sigma(iteration, max_iter) ** 2))
        return neighbourhood

    def _update_weights(self, x, bmu_index, iteration, max_iter):
        for i in range(self.m):
            for j in range(self.n):
                distance_to_bmu = self._distance_to_bmu(bmu_index, i, j)
                neighbourhood = self._neighbourhood(distance_to_bmu, iteration, max_iter)
                self.weights[i, j] += neighbourhood * self.learning_rate * (x - self.weights[i, j])

    def train(self, X, epochs):
        for iteration in range(epochs):
            for x in X:
                bmu_index = self._find_bmu(x)
                self._update_weights(x, bmu_index, iteration, epochs)

    def map_vecs(self, X) -> np.ndarray:
        mapped = []
        for x in X:
            bmu_index = self._find_bmu(x)
            mapped.append(bmu_index)
        map_v = np.array(mapped)
        return map_v
