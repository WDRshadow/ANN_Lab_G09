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