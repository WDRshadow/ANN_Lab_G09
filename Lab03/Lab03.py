import unittest
import numpy as np
import matplotlib.pyplot as plt

from utils import Read_Files

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
        x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
        x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
        x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1]).reshape(1, -1)

        x = np.array([x1, x2, x3])

        # hopfield = HopfieldNetwork(x)

        x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1]).reshape(1, -1)
        x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1]).reshape(1, -1)
        x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1]).reshape(1, -1)

        return

    def test_print(self):
        images = self.data_generator.data.reshape((11, 1024))

        print_image(images[2].reshape((32, 32)))

        

