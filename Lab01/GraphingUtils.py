import numpy as np
import matplotlib.pyplot as plt

def plot_config(ax, title="Generated Data"):
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2) 
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()

def plot_data(ax, class0, class1):
    """
    Plot the generated data for class 1 and class 0.
    """
    ax.scatter(class1[0, :], class1[1, :], color='blue', label='Class 1')
    ax.scatter(class0[0, :], class0[1, :], color='red', label='Class 0')

def plot_line_from_vector(class0, class1, direction_vector: np.ndarray, point=np.array([0, 0]), title="Generated Data"):
    """
    Plot a 2D line given a direction vector and a point.

    Parameters:
        direction_vector (np.ndarray): The direction vector [a, b] of the line.
        point (np.ndarray): A point [x0, y0] through which the line passes (default is the origin).
    """
    fig, ax = plt.subplots()

    ax.quiver(point[0], point[1], direction_vector[0], direction_vector[1], 
           angles='xy', scale_units='xy', scale=1, color='C0')
    try:
        slope = -direction_vector[0] / direction_vector[1]
    except ZeroDivisionError:
        slope = float('inf')

    ax.axline(xy1=point, slope=slope, color='C0', label='Decision Line')

    plot_data(ax, class0, class1)
    plot_config(ax, title)

    plt.show()
