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


def plot_line_from_vector(class0: np.ndarray, class1: np.ndarray, direction_vector: np.ndarray, point=np.array([0, 0]), title="Generated Data"):
    """
    Plot a 2D line given a direction vector and a point.

    Parameters:
        class0 (np.ndarray): Data points for class 0, shape (n, 2).
        class1 (np.ndarray): Data points for class 1, shape (n, 2).
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

    plot_data(ax, class0.transpose(), class1.transpose())
    plot_config(ax, title)

    plt.show()


def plot_line_from_weights(class0: np.ndarray, class1: np.ndarray, weights: np.ndarray, title="Perceptron Learning Data"):
    """
    Plot a 2D line given a weight vector.

    Parameters:
        weights (np.ndarray): The weight vector [w0, w1, w2] of the line. The line should be w0 + w1*x + w2*y = 0.
    """
    fig, ax = plt.subplots()

    # Plot the decision line
    x = np.linspace(-2, 2, 100)
    y = (-weights[0] / weights[2]) - (weights[1] / weights[2]) * x
    ax.plot(x, y, color='C0', label='Decision Line')

    plot_data(ax, class0.transpose(), class1.transpose())
    plot_config(ax, title)

    plt.show()
