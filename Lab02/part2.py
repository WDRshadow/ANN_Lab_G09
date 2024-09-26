import unittest

import matplotlib.pyplot as plt
import numpy as np

from utils import SOM_2D, Read_Files


class Lab02Part2DataGenerator:
    def __init__(self):
        self.animalattributes: list[str] = Read_Files("data_lab2/animalattributes.txt").read_lines()
        self.animalnames: list[str] = Read_Files("data_lab2/animalnames.txt").read_lines("'")
        self.animals: np.ndarray = Read_Files("data_lab2/animals.dat").read_one_line(type_=int)
        self.cities: np.ndarray = Read_Files("data_lab2/cities.dat").read_matrix(",", ";", float)
        self.mpdistrict: np.ndarray = Read_Files("data_lab2/mpdistrict.dat").read_matrix(" ", type_=int).T.reshape(-1)
        self.mpnames: list[str] = Read_Files("data_lab2/mpnames.txt", encoding="ISO-8859-1").read_lines()
        self.mpparty: np.ndarray = Read_Files("data_lab2/mpparty.dat").read_matrix(" ", type_=int).T.reshape(-1)
        self.mpsex: np.ndarray = Read_Files("data_lab2/mpsex.dat").read_matrix(" ", type_=int).T.reshape(-1)
        self.votes: np.ndarray = Read_Files("data_lab2/votes.dat").read_one_line(type_=float)


def animal_ordering():
    data = np.loadtxt('data_lab2/animals.dat', delimiter=",", dtype=int)
    data = np.reshape(data, (32, 84))

    # 100x1 grid, 20 epochs, 0.2 learning rate
    grid_size_m = 100
    grid_size_n = 1
    learning_rate = 0.2

    som = SOM_2D(grid_size_m, grid_size_n, 84, learning_rate, sigma_0=10)
    som.train(data, epochs=20)

    names = np.loadtxt('data_lab2/animalnames.txt', dtype=str)
    for i, name in enumerate(names):
        names[i] = name[1:-1]

    mapped = som.map_vecs(data)

    combined = list(zip(mapped, names))
    combined.sort(key=lambda x: (x[0][0], x[0][1]))
    sorted_mapped, sorted_names = zip(*combined)

    x_coords = [x[0] for x in sorted_mapped]
    y_coords = [y[1] for y in sorted_mapped]

    # Plot the positions
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_coords, y_coords, c='blue')
    plt.xlim(0, 100)
    plt.ylim(-1, 1)
    plt.scatter(x_coords, y_coords, c='blue')

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if i % 2 == 0:
            ax.annotate(sorted_names[i], (x, y), rotation=45,
                        va="bottom", ha="center", fontsize=9,
                        color='black', bbox=dict(facecolor='white', alpha=0.5))
        else:
            ax.annotate(sorted_names[i], (x, y), rotation=45,
                        va="top", ha="center", fontsize=9,
                        color='black', bbox=dict(facecolor='white', alpha=0.5))

    plt.title('Animal sorted by SOM as vector')
    plt.xlabel('Lattice axis')
    plt.show()


def cyclic_tour():
    with open('data_lab2/cities.dat', 'r') as file:
        content = file.read().replace(';', '\n')

    data = np.loadtxt(content.splitlines(), delimiter=",", dtype=float, skiprows=3)
    data = np.reshape(data, (10, 2))

    # 10x10 grid, 100 epochs, 0.3 learning rate
    grid_size_m = 10
    grid_size_n = 10
    learning_rate = 0.3

    som = SOM_2D(grid_size_m, grid_size_n, 2, learning_rate, sigma_0=3)
    som.train(data, epochs=100)

    mapped = som.map_vecs(data)

    x_mapped = [float(x[0]) / 10.0 for x in mapped]
    y_mapped = [float(y[1]) / 10.0 for y in mapped]

    # Mapped positions
    plt.figure(figsize=(10, 8))
    plt.plot(x_mapped, y_mapped, marker='o', color='blue', label='Mapped Positions', linestyle='-', linewidth=1)

    # Data points
    plt.scatter(data[:, 0], data[:, 1], c='red', label='Data Points', s=50)

    plt.title('Mapped Points and Data Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()


class TestSOM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSOM, self).__init__(*args, **kwargs)
        self.data_generator = Lab02Part2DataGenerator()

    def test_animal(self):
        # reshape the animal data to (32 * 84)
        animal_data = self.data_generator.animals.reshape((32, 84))
        animal_names = self.data_generator.animalnames
        # 3*sigma = 50
        som = SOM_2D(100, 1, 84, 0.2, 16.67)
        som.train(animal_data, epochs=20)
        map_pos = som.map_vecs(animal_data)
        map_pos[:, 1] = np.arange(32)
        map_pos = map_pos.tolist()
        map_pos.sort(key=lambda x: x[0])
        for animal in map_pos:
            print(f"Animal: {animal_names[int(animal[1])]}; Position: {animal[0]}")

    def test_city(self):
        city_pos = self.data_generator.cities

        class SOM_cycle(SOM_2D):
            def __init__(self, m, dim, learning_rate=0.3, sigma_0=0.67):
                super().__init__(m, 1, dim, learning_rate, sigma_0)

            def _distance_to_bmu(self, bmu_index, i, j):
                distance_main = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))
                distance_left = np.linalg.norm(np.array([i - self.m, j]) - np.array(bmu_index))
                distance_right = np.linalg.norm(np.array([i + self.m, j]) - np.array(bmu_index))
                return min(distance_main, distance_left, distance_right)

        # 3*sigma = 2
        som = SOM_cycle(10, 2, 0.3, 0.67)
        som.train(city_pos, epochs=400)
        map_index = som.map_vecs(city_pos)
        map_index[:, 1] = np.arange(10)
        map_index = map_index.tolist()
        map_index.sort(key=lambda x: x[0])

        # plot the cities with a line following the order
        city_pos_order = city_pos[np.array(map_index).T[1].tolist()]
        start_point = city_pos_order[0]
        city_pos_order = np.vstack((city_pos_order, start_point))
        X = city_pos_order[:, 0]
        Y = city_pos_order[:, 1]
        plt.plot(X, Y, marker='o', color='blue', label='Mapped Positions', linestyle='-', linewidth=1)
        plt.scatter(city_pos[:, 0], city_pos[:, 1], c='red', label='Data Points', s=50)
        plt.title('Mapped Points and Data Points')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_mp(self):
        # reshape the vote data to (349 * 31)
        votes = self.data_generator.votes.reshape((349, 31))
        # reshape the mp data to (349 * 3)
        mp_party = self.data_generator.mpparty.reshape((349, 1))
        mp_sex = self.data_generator.mpsex.reshape((349, 1))
        mp_district = self.data_generator.mpdistrict.reshape((349, 1))

        # Mapping the votes, with 3*sigma = 2
        vote_som = SOM_2D(10, 10, 31, 0.3, 0.67)
        vote_som.train(votes, epochs=100)
        vote_map = vote_som.map_vecs(votes)

        # plot the votes
        def plot_votes(mapped, mp_class, title="Mapped Points and Data Points"):
            """Plot the 2d map of the votes into a graph with points with different colors"""
            classes = np.unique(mp_class)
            num_class = len(classes)
            plt.title(title)
            for i in range(num_class):
                class_i = np.where(mp_class == classes[i])[0]
                plt.scatter(mapped[class_i, 0], mapped[class_i, 1], label=classes[i])
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True)
            if num_class < 5:
                plt.legend()
            plt.show()

        plot_votes(vote_map, mp_party, "Party")
        plot_votes(vote_map, mp_sex, "Gender")
        plot_votes(vote_map, mp_district, "District")

    def test_animals(self):
        animal_ordering()

    def test_cycling(self):
        cyclic_tour()
