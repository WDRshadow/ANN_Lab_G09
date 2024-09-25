import unittest

import numpy as np
import matplotlib.pyplot as plt

from utils import SOM

def animal_ordering():
    data = np.loadtxt('./Lab02/P2Data/animals.dat', delimiter=",", dtype=int)
    data = np.reshape(data, (32, 84))

    # 100x1 grid, 20 epochs, 0.2 learning rate
    grid_size_m = 100
    grid_size_n = 1
    learning_rate = 0.2

    som = SOM(grid_size_m, grid_size_n, 84, learning_rate, sigma_0=10)
    som.train(data, epochs=20)

    names = np.loadtxt('./Lab02/P2Data/animalnames.txt', dtype=str)
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
    with open('./Lab02/P2Data/cities.dat', 'r') as file:
        content = file.read().replace(';', '\n')

    data = np.loadtxt(content.splitlines(), delimiter=",", dtype=float, skiprows=3)
    data = np.reshape(data, (10, 2))

    # 10x10 grid, 100 epochs, 0.3 learning rate
    grid_size_m = 10
    grid_size_n = 10
    learning_rate = 0.3

    som = SOM(grid_size_m, grid_size_n, 2, learning_rate, sigma_0=3)
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
    # def test(self):
    #     data = np.random.rand(100, 3)
    #     som = SOM(10, 10, 3)
    #     som.train(data, epochs=100)
    #     mapped = som.map_vecs(data)
    #     print("Mapped position: ", mapped[:10])

    # def test_animals(self):
    #     animal_ordering()

    def test_cycling(self):
        cyclic_tour()