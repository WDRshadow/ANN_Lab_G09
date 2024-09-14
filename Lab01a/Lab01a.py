import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from sympy import false

from DataGenerator import DataGenerator
from utils import SingleLevelPerceptron, PerceptronLearning, DeltaRuleLearning


def plot(model: SingleLevelPerceptron, data: (np.ndarray, np.ndarray), title="Generated Data"):
    """
    Plot the data and the decision line
    :param model:  the model
    :param data:  the data
    :param title:  the title of the plot
    """
    fig, ax = plt.subplots()

    y_0 = -model.w[0] / model.w[2]
    point = np.array([0, y_0[0]])

    # plot the normal vector to the decision line at the point
    ax.quiver(*point, model.w[1], -model.w[2], color='C0', label='Normal Vector')

    try:
        slope = -model.w[1] / model.w[2]
        slope = slope[0]
    except ZeroDivisionError:
        slope = float('inf')

    ax.axline(xy1=point, slope=slope, color='C0', label='Decision Line')

    labels = data[1].flatten()
    class1 = data[0][labels == 1]
    class0 = data[0][labels == -1]

    ax.scatter(class1.transpose()[0, :], class1.transpose()[1, :], color='blue', label='Class 1')
    ax.scatter(class0.transpose()[0, :], class0.transpose()[1, :], color='red', label='Class 0')

    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()

    plt.show()


def plot_errors(model: SingleLevelPerceptron):
    """
    Plot the errors over the epochs
    """
    plt.plot(model.errors)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error over epochs")
    plt.show()


def two_in_one_plot(data_generator: DataGenerator, model1: SingleLevelPerceptron, model2: SingleLevelPerceptron):
    fig, ax = plt.subplots()

    y_0 = -model1.w[0] / model1.w[2]
    point = np.array([0, y_0[0]])

    # plot the normal vector to the decision line at the point
    ax.quiver(*point, model1.w[1], -model1.w[2], color='C0')

    try:
        slope = -model1.w[1] / model1.w[2]
        slope = slope[0]
    except ZeroDivisionError:
        slope = float('inf')

    ax.axline(xy1=point, slope=slope, color='C0', label='Decision Line' + model1.__class__.__name__)

    y_0 = -model2.w[0] / model2.w[2]
    point = np.array([0, y_0[0]])

    # plot the normal vector to the decision line at the point
    ax.quiver(*point, model2.w[1], -model2.w[2], color='C1')

    try:
        slope = -model2.w[1] / model2.w[2]
        slope = slope[0]
    except ZeroDivisionError:
        slope = float('inf')

    ax.axline(xy1=point, slope=slope, color='C1', label='Decision Line' + model2.__class__.__name__)

    labels = data_generator.data[1].flatten()
    class1 = data_generator.data[0][labels == 1]
    class0 = data_generator.data[0][labels == -1]

    ax.scatter(class1.transpose()[0, :], class1.transpose()[1, :], color='blue', label='Class 1')
    ax.scatter(class0.transpose()[0, :], class0.transpose()[1, :], color='red', label='Class 0')

    ax.set_xlim(-2, 5)
    ax.set_ylim(-2, 5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title("Generated Data")
    ax.legend()

    plt.show()


def error_two_in_one_plot(model1: SingleLevelPerceptron, model2: SingleLevelPerceptron):
    plt.plot(model1.errors, color='C0', label=model1.__class__.__name__)
    plt.plot(model2.errors, color='C1', label=model2.__class__.__name__)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error over epochs")
    plt.legend()
    plt.show()


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.study_rate = 0.001
        self.epochs = 3000

        self.data_generator = DataGenerator(
            n=100,
            mA=[0.0, 2.0],
            sigmaA=0.3,
            mB=[-0.0, -0.0],
            sigmaB=0.3
        )

    def test_perceptron_learning(self):
        perceptron = PerceptronLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        perceptron.train(self.data_generator.data)
        plot(perceptron, self.data_generator.data, title="Perceptron Learning Data")
        plot_errors(perceptron)

    def test_delta_rule_learning(self):
        delta_rule = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule.train(self.data_generator.data, learning_type="b")
        plot(delta_rule, self.data_generator.data, title="Delta Rule Learning Data")
        plot_errors(delta_rule)

    def test_all(self, is_plot=True):
        perceptron = PerceptronLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        perceptron.train(self.data_generator.data)
        delta_rule = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule.train(self.data_generator.data, learning_type="b")
        if is_plot:
            two_in_one_plot(self.data_generator, perceptron, delta_rule)
            error_two_in_one_plot(perceptron, delta_rule)
        return perceptron, delta_rule

    def test_left_right_side(self):
        self.data_generator = DataGenerator(
            n=10,
            mA=[-1.0, 0.0],
            sigmaA=0.5,
            mB=[1.0, 0.0],
            sigmaB=0.5
        )
        self.test_all()

    def test_different_study_rate(self):
        perceptron, delta_rule = self.test_all(is_plot=false)
        self.study_rate = 0.1
        perceptron1, delta_rule1 = self.test_all(is_plot=false)
        two_in_one_plot(self.data_generator, perceptron, perceptron1)
        error_two_in_one_plot(perceptron, perceptron1)
        two_in_one_plot(self.data_generator, delta_rule, delta_rule1)
        error_two_in_one_plot(delta_rule, delta_rule1)

    def test_learning_type(self):
        # calculate systemds time
        time_0 = time.time()
        delta_rule = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule.train(self.data_generator.data, learning_type="s")
        print("Time for sequential: ", time.time() - time_0)
        time_0 = time.time()
        delta_rule2 = DeltaRuleLearning(2, 1, study_rate=self.study_rate, epochs=self.epochs)
        delta_rule2.train(self.data_generator.data, learning_type="b")
        print("Time for batch: ", time.time() - time_0)
        two_in_one_plot(self.data_generator, delta_rule, delta_rule2)
        error_two_in_one_plot(delta_rule, delta_rule2)
