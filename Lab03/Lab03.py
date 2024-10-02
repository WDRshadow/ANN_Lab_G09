import unittest
import numpy as np
import matplotlib.pyplot as plt

from utils import Read_Files
from utils import HopfieldNetwork

import itertools

class Lab03DataGenerator:
    def __init__(self):
        base_path = "./Lab03/data_lab3"
        self.data: np.ndarray = Read_Files(base_path + "/pict.dat").read_one_line(type_=int)

def print_image(image):
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()

class TestHopfield(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestHopfield, self).__init__(*args, **kwargs)
        self.data_generator = Lab03DataGenerator()

    def test_31(self):
        x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
        x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
        x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

        x = np.array([x1, x2, x3])

        hopfield = HopfieldNetwork(8)
        hopfield.train(x)

        x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1])
        x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1])
        x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1])

        for i, new_pattern in enumerate([x1d, x2d, x3d]):
            recalled_pattern = hopfield.recall(new_pattern, steps=2)

            print(f"Test pattern: \t\t{new_pattern}")
            print(f"Recalled pattern: \t{recalled_pattern} == x{i+1}? {np.all(np.equal(x[i], recalled_pattern))}")

            comparison_result = np.where(new_pattern == recalled_pattern, 1, 0)
            print(f"Comparisson: \t\t{comparison_result}\n")

        patterns = list(itertools.product([-1, 1], repeat=8))
        unique_recalled_patterns = set()

        for pattern in patterns:
            recalled_pattern = hopfield.recall(pattern)
            unique_recalled_patterns.add(tuple(recalled_pattern))
        unique_recalled_patterns = list(unique_recalled_patterns)

        print(len(unique_recalled_patterns), unique_recalled_patterns)

        x1crazy = np.array([-1, -1, 1, -1, -1, 1, 1, -1]) # 4 diff
        x2crazy = np.array([-1, -1, -1, 1, 1, -1, 1, 1]) # 5 diff
        x3crazy = np.array([-1, -1, -1, 1, 1, -1, 1, -1]) # 7 diff

        for i, new_pattern in enumerate([x1crazy, x2crazy, x3crazy]):
            recalled_pattern = hopfield.recall(new_pattern)

            print(f"Test pattern: \t\t{new_pattern}")
            print(f"Recalled pattern: \t{recalled_pattern} == x{i+1}inv? {np.all(np.not_equal(x[i], recalled_pattern))}")

            comparison_result = np.where(new_pattern == recalled_pattern, 1, 0)
            print(f"Comparisson: \t\t{comparison_result}\n")

        x1half = np.array([-1, -1, 1, -1, -1, 1, 1, -1]) # 4 diff
        x1half2 = np.array([1, 1, -1, 1, 1, -1, -1, 1]) # 4 diff

        for i, new_pattern in enumerate([x1half, x1half2]):
            recalled_pattern = hopfield.recall(new_pattern)

            print(f"Test pattern: \t\t{new_pattern}")
            print(f"Recalled pattern: \t{recalled_pattern} == x1inv? {np.all(np.not_equal(x1, recalled_pattern))}")

            comparison_result = np.where(new_pattern == recalled_pattern, 1, 0)
            print(f"Comparisson: \t\t{comparison_result}\n")

        x1half = np.array([1, -1, 1, -1, -1, 1, 1, -1]) # 5 diff
        x1half2 = np.array([1, 1, -1, 1, 1, -1, -1, -1]) # 5 diff

        for i, new_pattern in enumerate([x1half, x1half2]):
            recalled_pattern = hopfield.recall(new_pattern)

            print(f"Test pattern: \t\t{new_pattern}")
            print(f"Recalled pattern: \t{recalled_pattern} == x1inv? {np.all(np.not_equal(x1, recalled_pattern))}")

            comparison_result = np.where(new_pattern == recalled_pattern, 1, 0)
            print(f"Comparisson: \t\t{comparison_result}\n")

    def test_32(self):
        images = self.data_generator.data.reshape((11, 1024))
        print_image(images[9].reshape(32,32))

        hopfield = HopfieldNetwork(1024)

        hopfield.train(images[:3])

        recalled_pattern = hopfield.recall(images[9], steps=100, asyncronous=False)
        print_image(recalled_pattern.reshape(32,32))

        recalled_pattern = hopfield.recall(images[9], steps=100, random=True, asyncronous=False)
        print_image(recalled_pattern.reshape(32,32))

        print_image(images[1].reshape(32,32))
        print_image(images[2].reshape(32,32))

        recalled_pattern = hopfield.recall(images[10], steps=100, asyncronous=False)
        print_image(recalled_pattern.reshape(32,32))

        recalled_pattern = hopfield.recall(images[10], steps=100, random=True, asyncronous=False)
        print_image(recalled_pattern.reshape(32,32))


    def test_print(self):
        images = self.data_generator.data.reshape((11, 1024))

        print_image(images[2].reshape((32, 32)))


    def test_33_1(self):
        images = self.data_generator.data.reshape((11, 1024))
        p1 = images[0]
        p2 = images[1]
        p3 = images[2]
        p10 = images[9]
        p11 = images[10]

        hopfield = HopfieldNetwork(1024)
        hopfield.train(images[:3])

        print_image(p1.reshape(32,32))
        
        energy_p1 = hopfield.energy(p1)
        energy_p2 = hopfield.energy(p2)
        energy_p3 = hopfield.energy(p3)
        energy_p10 = hopfield.energy(p10)
        energy_p11 = hopfield.energy(p11)

        print(energy_p1)
        print(energy_p2)
        print(energy_p3)
        print(energy_p10)
        print(energy_p11)

        # recalled_p1 = hopfield.recall(p1, steps = 100)
        recalled_p10, energies_p10 = hopfield.recall(p10, steps = 3, energy_list=True)
        recalled_p11, energies_p11 = hopfield.recall(p11, steps = 3, energy_list=True)

        # print_image(recalled_p1.reshape(32,32))
        print_image(recalled_p10.reshape(32,32))
        print_image(recalled_p11.reshape(32,32))

        iterations = range(len(energies_p11)) 
        plt.plot(iterations, energies_p10, label="p10", color='green', linestyle='--', marker='x')
        plt.plot(iterations, energies_p11, label="p11", color='red', linestyle='-.', marker='s')
        plt.xlabel('Iterations')
        plt.ylabel('Energy')
        plt.title('Energy vs. Iterations')
        plt.legend()
        plt.show()


    def test_33_2(self):
        images = self.data_generator.data.reshape((11, 1024))
        p1 = images[0]
        p2 = images[1]
        p3 = images[2]
        p10 = images[9]
        p11 = images[10]

        arbitrary_start = np.random.choice([1, -1], (1024,1))

        hopfield_1 = HopfieldNetwork(1024)
        hopfield_1.weights = hopfield_1.initialize_weights(random_gauss=True, symmetric=False)
        hopfield_2 = HopfieldNetwork(1024)
        hopfield_2.weights = np.multiply(0.5, np.add(hopfield_1.weights, hopfield_1.weights.T))
        # hopfield_2.weights = hopfield_2.initialize_weights(random_gauss=True, symmetric=True)
        hopfield_1.train(images[:3])
        hopfield_2.train(images[:3])



        recalled_1, energies_1 = hopfield_1.recall(arbitrary_start, steps = 3, energy_list=True)
        recalled_2, energies_2 = hopfield_2.recall(arbitrary_start, steps = 3, energy_list=True)

        if np.array_equal(recalled_1, recalled_2):
            print("equal")
        else:
            print("notequal")

        iterations = range(len(energies_1)) 
        plt.plot(iterations, energies_1, label="Non symmetric", color='green')
        plt.plot(iterations, energies_2, label="Symmetric", color='red')
        plt.xlabel('Iterations')
        plt.ylabel('Energy')
        plt.title('Energy vs. Iterations')
        plt.legend()
        plt.show()


    def test_34(self):
        images = self.data_generator.data.reshape((11, 1024))
        p1 = images[0]
        p2 = images[1]
        p3 = images[2]
        hopfield = HopfieldNetwork(1024)
        hopfield.train(images[:3])

        def add_noise(pattern, num_flip):
            noisy_pattern = pattern.copy()
            flip_indices = np.random.choice(1024, num_flip, replace=False)
            noisy_pattern[flip_indices] *= -1
            return noisy_pattern

        p1_rate = {round(i, 1): 0 for i in [x * 0.1 for x in range(11)]}
        p2_rate = {round(i, 1): 0 for i in [x * 0.1 for x in range(11)]}
        p3_rate = {round(i, 1): 0 for i in [x * 0.1 for x in range(11)]}

        for i in range(50):
            for noise_level in np.arange(0, 1.1, 0.1):
                print(noise_level)
                noise_level = round(noise_level, 1)
                num_flip = int(noise_level * 1024)

                noisy_p1 = add_noise(p1, num_flip)
                noisy_p2 = add_noise(p2, num_flip)
                noisy_p3 = add_noise(p3, num_flip)

                recp1 = hopfield.recall(noisy_p1, asyncronous=False, steps=10)
                recp2 = hopfield.recall(noisy_p2, asyncronous=False, steps=10)
                recp3 = hopfield.recall(noisy_p3, asyncronous=False, steps=10)

                if np.array_equal(p1, recp1):
                    p1_rate[noise_level] = p1_rate[noise_level] + 1

                if np.array_equal(p2, recp2):
                    p2_rate[noise_level] = p2_rate[noise_level] + 1

                if np.array_equal(p3, recp3):
                    p3_rate[noise_level] = p3_rate[noise_level] + 1


        noise_levels = list(p1_rate.keys()) 
        p1_values = list(p1_rate.values())
        p2_values = list(p2_rate.values())
        p3_values = list(p3_rate.values())
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, p1_values, marker='o', label='Pattern 1')
        plt.plot(noise_levels, p2_values, marker='o', label='Pattern 2')
        plt.plot(noise_levels, p3_values, marker='o', label='Pattern 3')

        plt.title('Number of Correct Recall vs Noise Level with step equal to ten')
        plt.xlabel('Noise Level')
        plt.ylabel('Number of Correct Recall')
        plt.xticks(noise_levels)
        plt.grid()
        plt.legend()
        plt.show()