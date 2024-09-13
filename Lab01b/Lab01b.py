import unittest

from DataGenerator import DataGenerator
from Module import Module, Layer, Sigmoid


class MLP(Module):
    def __init__(self, input_dim: int, output_dim: int, study_rate=0.001, epochs=3000):
        super().__init__(study_rate, epochs)
        self.layer1 = Layer(input_dim, 3, Sigmoid)
        self.layer2 = Layer(3, output_dim, Sigmoid)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.study_rate = 0.001
        self.epochs = 3000

        self.data_generator = DataGenerator(
            n=100,
            mA=[1.0, 0.3],
            sigmaA=0.2,
            mB=[-0.0, -0.1],
            sigmaB=0.3
        )

    def test(self):
        # self.data_generator.plot()
        mlp = MLP(2, 1, self.study_rate, self.epochs)
        X, Y = self.data_generator.data
        mlp.train(X, Y, msg=True)
