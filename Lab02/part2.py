import unittest

import numpy as np

from utils import SOM, Read_Files


class Lab02Part2DataGenerator:
    def __init__(self):
        self.animalattributes: list[str] = Read_Files("data_lab2/animalattributes.txt").read_lines()
        self.animalnames: list[str] = Read_Files("data_lab2/animalnames.txt").read_lines("'")
        self.animals: np.ndarray = Read_Files("data_lab2/animals.dat").read_one_line(type_=int)
        self.ballist: np.ndarray = Read_Files("data_lab2/ballist.dat").read_matrix(" ", "\t", float)
        self.balltest: np.ndarray = Read_Files("data_lab2/balltest.dat").read_matrix(" ", "\t", float)
        self.cities: np.ndarray = Read_Files("data_lab2/cities.dat").read_matrix(",", ";", float)
        self.mpdistrict: np.ndarray = Read_Files("data_lab2/mpdistrict.dat").read_matrix(" ", type_=int).T.reshape(-1)
        self.mpnames: list[str] = Read_Files("data_lab2/mpnames.txt", encoding="ISO-8859-1").read_lines()
        self.mpparty: np.ndarray = Read_Files("data_lab2/mpparty.dat").read_matrix(" ", type_=int).T.reshape(-1)
        self.mpsex: np.ndarray = Read_Files("data_lab2/mpsex.dat").read_matrix(" ", type_=int).T.reshape(-1)
        self.votes: np.ndarray = Read_Files("data_lab2/votes.dat").read_one_line(type_=float)


class TestSOM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSOM, self).__init__(*args, **kwargs)
        self.data_generator_animal = None
        self.data_generator_cities = None
        self.data_generator_mp = None

    def test_data_reader(self):
        data = Lab02Part2DataGenerator()
        print("Animal attributes: ", data.animalattributes[:10])
        print("Animal names: ", data.animalnames[:10])
        print("Animals: ", data.animals[:10])
        print("Ballist: ", data.ballist[:10])
        print("Balltest: ", data.balltest[:10])
        print("Cities: ", data.cities[:10])
        print("MP District: ", data.mpdistrict[:10])
        print("MP Names: ", data.mpnames[:10])
        print("MP Party: ", data.mpparty[:10])
        print("MP Sex: ", data.mpsex[:10])
        print("Votes: ", data.votes[:10])


    def test(self):
        data = np.random.rand(100, 3)
        som = SOM(10, 10, 3)
        som.train(data, epochs=100)
        mapped = som.map_vecs(data)
        print("Mapped position: ", mapped[:10])