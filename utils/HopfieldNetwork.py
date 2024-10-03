import numpy as np

class HopfieldNetwork:
    def __init__(self, n_units):
        self.n_units = n_units
        self.weights = np.zeros((n_units, n_units))

    def initialize_weights(self, random_gauss=False, symmetric=True):
        if random_gauss:
            w = np.random.normal(0, 1, (self.n_units, self.n_units))
            np.fill_diagonal(w, 0)
            if symmetric:
                w = np.multiply(0.5, np.add(w, w.T))
            return w

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

        # print("!!!!!!!!!!!!")
        # print(self.weights)

    def update(self, state, steps=1, random=False, asyncronous=True, energy_list=False, stable=False):
        aux_state = np.copy(state)
        energies = [self.energy(aux_state)]
        is_stable = False
        for _ in range(steps):
            indices = range(self.n_units)
            if random: 
                indices = np.random.permutation(self.n_units)

            if asyncronous:
                for i in indices:
                    raw_input = np.dot(self.weights[i], state)

                    state[i] = 1 if raw_input >= 0 else -1
                    if energy_list:
                        energies.append(self.energy(state))
            else:
                for i in indices:
                    raw_input = np.dot(self.weights[i], state)
                    aux_state[i] = 1 if raw_input >= 0 else -1
                state = np.copy(aux_state)
                energies.append(self.energy(state))
                
        if energy_list:
            return state, energies
        return state

    def recall(self, pattern, steps=10, random=False, asyncronous=True, energy_list=False):
        state = np.copy(pattern)
        return self.update(state, steps, random, asyncronous, energy_list)
    
    # def energy(self, state):
    #     product = np.outer(state, np.transpose(state))
    #     energy_sum = np.sum(np.multiply(self.weights, product))
    #     return -energy_sum
    
    def energy(self, state):
        """Calculate the energy of the current state."""
        return -0.5 * np.sum(np.dot(state, np.dot(self.weights, state)))

    def count_stable_patterns(self, patterns):
        stable_count = 0
        for pattern in patterns:
            recalled = self.recall(pattern)
            if np.array_equal(recalled, pattern):
                stable_count += 1
        return stable_count