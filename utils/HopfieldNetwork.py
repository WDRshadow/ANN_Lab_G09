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

    def update(self, state, steps=1, random=False, asyncronous=True):
        aux_state = state
        print(asyncronous)
        for _ in range(steps):
            indices = range(self.n_units)
            if random: indices = np.random.permutation(self.n_units)

            if asyncronous:
                for i in indices:
                    raw_input = np.dot(self.weights[i], state)
                    state[i] = 1 if raw_input >= 0 else -1
            else:
                for i in indices:
                    raw_input = np.dot(self.weights[i], state)
                    aux_state[i] = 1 if raw_input >= 0 else -1
                state = aux_state
        return state

    def recall(self, pattern, steps=10, random=False, asyncronous=True):
        state = np.copy(pattern)
        return self.update(state, steps, random, asyncronous)