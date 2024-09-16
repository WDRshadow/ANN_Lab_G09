import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import seaborn as sns

from Lab01b.part1 import plot_loss, plot_losses
from utils import MackeyGlass

class MLP(nn.Module):
    def __init__(self, h1: int = 5, h2: int = 6):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.Sigmoid(self.fc1(x))
        x = self.Sigmoid(self.fc2(x))
        x = self.Sigmoid(self.fc3(x))
        return x


def train(model: nn.Module, train_X, train_Y, val_X, val_Y, epochs=3000, study_rate=0.001, regularization=None,
          patience=10, msg=True):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=study_rate, weight_decay=regularization if regularization else 0.0)

    best_val_loss = float('inf')
    patience_counter = 0
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_X)
        loss = criterion(output, train_Y)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_X)
            val_loss = criterion(val_output, val_Y)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        if epoch % 100 == 0 and msg:
            print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
        losses.append(loss.item())
    return losses


def test(model, test_X, test_Y):
    model.eval()
    with torch.no_grad():
        predictions = model(test_X).reshape(-1)
    test_loss = torch.mean(torch.square(predictions - test_Y))
    print(f'Test Loss: {test_loss}')
    return test_loss.item()



def graph_matrix(items, n1, n2, labelx='n1 (Number of Nodes in First Layer)', labely='n2 (Number of Nodes in Second Layer)', title='Test Loss for Different Node Combinations (n1 vs n2)'):
    accuracy_matrix = np.array(items).reshape(len(n1), len(n2))

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(accuracy_matrix, annot=True, cmap="YlGnBu", xticklabels=n2, yticklabels=n1, fmt='.6f')
    ax.invert_yaxis()

    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    ax.set_title(title)

    plt.show()


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.data_generator = MackeyGlass()
        self.data_generator.generate_data()
        self.X_test, self.Y_test = self.data_generator.randomly_split_data(5/6)

    def train_and_test(self, model: MLP = None, regularization=0.001, epochs=30000, study_rate=0.001, is_plot=True, train_percentage=0.8):
        if model is None:
            model = MLP()
        X_val, Y_val = self.data_generator.randomly_split_data(train_percentage)
        X_train, Y_train = self.data_generator.data
        losses = train(model, torch.tensor(X_train).float(), torch.tensor(Y_train).float(),
                       torch.tensor(X_val).float(), torch.tensor(Y_val).float(), epochs, study_rate,
                       regularization, msg=False)
        if is_plot:
            plot_loss(losses)
        accuracy = test(model, torch.tensor(self.X_test).float(), torch.tensor(self.Y_test).float())
        return losses, accuracy

    def test(self):
        self.train_and_test()

    def test_dim_cases(self):
        losses = []
        test_losses = []
        n1 = [3, 4, 5]
        n2 = [2, 4, 6]
        for h1 in n1:
            for h2 in n2:
                print(f'Testing with h1={h1}, h2={h2}')
                loss, accuracy = self.train_and_test(MLP(h1, h2), is_plot=False)
                losses.append(loss)
                test_losses.append(accuracy)
        plot_losses(losses, ' - Hidden Dimension', 'line ')
        graph_matrix(test_losses, n1, n2)

    def test_gaussian_noise(self):
        losses = []
        test_losses = []
        n2 = [2, 6, 9]
        noice = [0.05, 0.15]
        for n in noice:
            for h2 in n2:
                print(f'Testing with noise={n}, h2={h2}')
                self.data_generator.generate_data()
                self.data_generator.add_gaussian_noise(std=n)
                loss, accuracy = self.train_and_test(MLP(h2=h2), is_plot=False)
                losses.append(loss)
                test_losses.append(accuracy)
        plot_losses(losses, ' - Gaussian Noise', 'line ')
        graph_matrix(test_losses, noice, n2, 'Noise Standard Deviation', 'n2 (Number of Nodes in Second Layer)', 'Test Loss for Different Noise Levels (Noise vs n2)')
