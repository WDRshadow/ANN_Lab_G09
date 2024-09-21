import unittest

import numpy as np

from utils import SOM

class TestSOM(unittest.TestCase):
    def test(self):
        data = np.random.rand(100, 3)
        som = SOM(10, 10, 3)
        som.train(data, epochs=100)
        mapped = som.map_vecs(data)
        print("Mapped position: ", mapped[:10])