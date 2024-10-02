import numpy as np

class HopfieldNetwork:
    def __init__(self, n_units):
        self.n_units = n_units
        self.weights = np.zeros((n_units, n_units))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def update(self, state, steps=1):
        for _ in range(steps):
            for i in range(self.n_units):
                raw_input = np.dot(self.weights[i], state)
                state[i] = 1 if raw_input >= 0 else -1
        return state

    def recall(self, pattern, steps=10):
        state = np.copy(pattern)
        return self.update(state, steps)