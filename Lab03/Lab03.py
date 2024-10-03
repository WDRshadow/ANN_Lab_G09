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
        plt.plot(iterations, energies_p10, label="Pattern 10", color='green')
        plt.plot(iterations, energies_p11, label="Pattern 11", color='red')
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

        

        arbitrary_start1 = np.random.choice([-1, 1], 1024)
        arbitrary_start2 = np.random.choice([-1, 1], 1024)

        hopfield_1 = HopfieldNetwork(1024)
        hopfield_1.weights = hopfield_1.initialize_weights(random_gauss=True, symmetric=False)
        hopfield_2 = HopfieldNetwork(1024)
        hopfield_2.weights = hopfield_2.initialize_weights(random_gauss=True, symmetric=True)
        hopfield_1.train(images[:3])
        hopfield_2.train(images[:3])

        print(hopfield_1.weights)
        print(hopfield_2.weights)
        print(arbitrary_start1)
        print(arbitrary_start2)

        recalled_1, energies_1 = hopfield_1.recall(arbitrary_start1, steps = 3, energy_list=True, asyncronous=True, random=True)
        recalled_2, energies_2 = hopfield_2.recall(arbitrary_start2, steps = 3, energy_list=True, asyncronous=True, random=True)

        print_image(recalled_1.reshape(32,32))
        print_image(recalled_2.reshape(32,32))
        print_image(p10.reshape(32,32))

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


    def add_noise(self, pattern, num_flip):
        noisy_pattern = pattern.copy()
        flip_indices = np.random.choice(100, num_flip, replace=False)
        noisy_pattern[flip_indices] *= -1
        return noisy_pattern

    def test_34(self):
        images = self.data_generator.data.reshape((11, 1024))
        p1 = images[0]
        p2 = images[1]
        p3 = images[2]
        hopfield = HopfieldNetwork(1024)
        hopfield.train(images[:3])

        p1_rate = {round(i, 1): 0 for i in [x * 0.1 for x in range(11)]}
        p2_rate = {round(i, 1): 0 for i in [x * 0.1 for x in range(11)]}
        p3_rate = {round(i, 1): 0 for i in [x * 0.1 for x in range(11)]}

        for i in range(50):
            for noise_level in np.arange(0, 1.1, 0.1):
                print(noise_level)
                noise_level = round(noise_level, 1)
                num_flip = int(noise_level * 1024)

                noisy_p1 = self.add_noise(p1, num_flip)
                noisy_p2 = self.add_noise(p2, num_flip)
                noisy_p3 = self.add_noise(p3, num_flip)

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

    def test_35(self):
        # add patterns p4, p5, p6, p7 in the network and see if distorted versions can be recognized

        images = self.data_generator.data.reshape((11, 1024))

        noise_list = {0: [], 10: [], 20: []}

        for i in range(10):
            hopfield = HopfieldNetwork(1024)
            hopfield.train(images[:i + 1])

            for noise_level in [0, 0.1, 0.2]:
                noise_level = round(noise_level, 1)
                num_flip = int(noise_level * 1024)

                correct_recall_count = 0
                for j in range(i + 1):
                    images_noisy = self.add_noise(images[j], num_flip)
                    recp1 = hopfield.recall(images_noisy, steps=3)

                    if np.array_equal(images[j], recp1):
                        correct_recall_count += 1

                recall_ratio = correct_recall_count / (i + 1)
                noise_list[int(noise_level * 100)].append(recall_ratio)

        x = range(1,11)  # Number of patterns
        # print(noise_list)
        plt.plot(x, noise_list[0], label="0% Noise")
        plt.plot(x, noise_list[10], label="10% Noise")
        plt.plot(x, noise_list[20], label="20% Noise")

        plt.xlabel('Number of Patterns')
        plt.ylabel('Recall Ratio')
        plt.title('Recall Ratio vs Number of Patterns at Different Noise Levels')
        plt.xticks(x)
        plt.legend()
        plt.grid(True)
        plt.show()


    def test_35_similarity(self):
        images = self.data_generator.data.reshape((11, 1024))

        images = images[:10]

        similarity_matrix = np.zeros((10, 10))

        for i in range(10):
            for j in range(10):
                similarity_matrix[i, j] = np.sum(images[i] == images[j]) / 1024

        print("Similarity Table (Dot Product):")
        for row in similarity_matrix:
            print("  ".join(f"{val:.2f}" for val in row))

    def test_35_random_samples(self):
        noise_list = {0: [], 10: [], 20: []}

        images = np.random.choice([1, -1], size=(150, 1024))

        for i in range(0, 150, 10):
            print(i)
            hopfield = HopfieldNetwork(1024)
            hopfield.train(images[:i + 1])

            for noise_level in [0, 0.1, 0.2]:
                noise_level = round(noise_level, 1)
                num_flip = int(noise_level * 1024)

                correct_recall_count = 0
                for j in range(i + 1):
                    images_noisy = self.add_noise(images[j], num_flip)
                    recp1 = hopfield.recall(images_noisy, steps=3)

                    if np.array_equal(images[j], recp1):
                        correct_recall_count += 1

                recall_ratio = correct_recall_count / (i + 1)
                noise_list[int(noise_level * 100)].append(recall_ratio)

        x = range(1, 151, 10)  # Number of patterns
        # print(noise_list)
        plt.plot(x, noise_list[0], label="0% Noise")
        plt.plot(x, noise_list[10], label="10% Noise")
        plt.plot(x, noise_list[20], label="20% Noise")

        plt.xlabel('Number of Patterns')
        plt.ylabel('Recall Ratio')
        plt.title('Recall Ratio vs Number of Patterns at Different Noise Levels')
        plt.xticks(range(1, 151, 10))
        plt.legend()
        plt.grid(True)
        plt.show()


    def test_35_2(self):
        num_patterns = 300
        pattern_size = 100
        patterns = np.random.choice([1, -1], size=(num_patterns, pattern_size))

        hopfield = HopfieldNetwork(pattern_size)
        stable_counts = []

        for i in range(num_patterns):
            print(i)
            hopfield.train([patterns[i]]) 
            stable_count = hopfield.count_stable_patterns(patterns[:i+1])
            stable_counts.append(stable_count)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_patterns + 1), stable_counts, label='Stable Patterns', color='blue')
        plt.xlabel('Number of Learned Patterns')
        plt.ylabel('Number of Stable Patterns')
        plt.title('Stability of Patterns in Hopfield Network')
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_35_3(self):

        noise_list = {0: [], 10: [], 20: []}
        num_patterns = 300
        pattern_size = 100
        images = np.random.choice([1, -1], size=(num_patterns, pattern_size))

        hopfield = HopfieldNetwork(pattern_size)

        for i in range(0, 300, 10):
            print(i)
            hopfield = HopfieldNetwork(pattern_size)
            hopfield.train(images[:i + 1])

            for noise_level in [0, 0.1, 0.2]:
                noise_level = round(noise_level, 1)
                num_flip = int(noise_level * 100)

                correct_recall_count = 0
                for j in range(i + 1):
                    images_noisy = self.add_noise(images[j], num_flip)
                    recp1 = hopfield.recall(images_noisy, steps=3)

                    if np.array_equal(images[j], recp1):
                        correct_recall_count += 1

                recall_ratio = correct_recall_count / (i + 1)
                noise_list[int(noise_level * 100)].append(recall_ratio)

        x = range(1, 300, 10)  # Number of patterns
        # print(noise_list)
        plt.plot(x, noise_list[0], label="0% Noise")
        plt.plot(x, noise_list[10], label="10% Noise")
        plt.plot(x, noise_list[20], label="20% Noise")

        plt.xlabel('Number of Patterns')
        plt.ylabel('Recall Ratio')
        plt.title('Recall Ratio vs Number of Patterns at Different Noise Levels')
        plt.xticks(range(1, 300, 10))
        plt.legend()
        plt.grid(True)
        plt.show()

