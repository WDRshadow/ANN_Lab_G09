import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import quiver


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


def plot_line_from_vector(class0: np.ndarray, class1: np.ndarray, direction_vector: np.ndarray, title="Generated Data"):
    """
    Plot a 2D line given a direction vector and a point.

    The line should be 0 = bias + a*x1 + b*x2.

    Parameters:
        class0 (np.ndarray): Data points for class 0, shape (n, 2).
        class1 (np.ndarray): Data points for class 1, shape (n, 2).
        direction_vector (np.ndarray): The direction vector [bias, a, b] of the line.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots()

    y_0 = -direction_vector[0] / direction_vector[2]
    point = np.array([0, y_0[0]])

    # plot the normal vector to the decision line at the point
    ax.quiver(*point, direction_vector[1], -direction_vector[2], color='C0', label='Normal Vector')

    try:
        slope = -direction_vector[1] / direction_vector[2]
        slope = slope[0]
    except ZeroDivisionError:
        slope = float('inf')

    ax.axline(xy1=point, slope=slope, color='C0', label='Decision Line')

    plot_data(ax, class0.transpose(), class1.transpose())
    plot_config(ax, title)

    plt.show()
