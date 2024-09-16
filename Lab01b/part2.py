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
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_X)
        loss = criterion(output, train_Y)
        loss.backward()
        optimizer.step()

        # Validation phase
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

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
    return train_losses, val_losses

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
        self.X_test, self.Y_test = self.data_generator.randomly_pop_data(1/6, is_permanent=True)

    def train_and_test(self, model: MLP = None, regularization=0.001, epochs=30000, study_rate=0.001, is_plot=True, train_percentage=0.8):
        if model is None:
            model = MLP()
        self.data_generator.reset_data()
        X_val, Y_val = self.data_generator.randomly_pop_data(1 - train_percentage)
        X_train, Y_train = self.data_generator.data
        losses, val_losses = train(model, torch.tensor(X_train).float(), torch.tensor(Y_train).float(),
                       torch.tensor(X_val).float(), torch.tensor(Y_val).float(), epochs, study_rate,
                       regularization, msg=False)
        if is_plot:
            plot_losses([losses, val_losses], label=['Training Error','Validation Error'])
        accuracy = test(model, torch.tensor(self.X_test).float(), torch.tensor(self.Y_test).float())
        return losses, accuracy, val_losses

    def test(self):
        self.train_and_test()

    def test_dim_cases(self):
        losses = []
        test_losses = []
        val_test_losses = []
        order_refference = []
        n1 = [3, 4, 5]
        n2 = [2, 4, 6]
        for h1 in n1:
            for h2 in n2:
                print(f'Testing with h1={h1}, h2={h2}')
                train_loss, test_loss, val_loss = self.train_and_test(MLP(h1, h2), is_plot=False)
                losses.append(train_loss)
                test_losses.append(test_loss)
                val_test_losses.append(val_loss[-1])
                order_refference.append(f'n1:{h1} & n2:{h2}')
        plot_losses(losses, ' - Hidden Dimension', order_refference)
        graph_matrix(val_test_losses, n1, n2, 
                     title='Validation Error for Different Node Combinations (n1 vs n2)')
        graph_matrix(test_losses, n1, n2, 
                     title='Test Error for Different Node Combinations (n1 vs n2)')

    def test_training_percentage_case(self):
        losses = []
        test_losses = []
        percentages = [0.2, 0.4, 0.6, 0.8]
        for p in percentages:
            print(f'Testing with training percentage={p}')
            train_loss, test_loss, _ = self.train_and_test(train_percentage=p, is_plot=False)
            losses.append(train_loss)
            test_losses.append(test_loss)
        plot_losses(losses, ' - Training Percentage', 'line ')

    def test_gaussian_noise(self):
        losses = []
        test_losses = []
        order_refference = []
        n2 = [3, 6, 9]
        noise = [0.05, 0.15]
        for n in noise:
            self.data_generator.reset_data()
            self.data_generator.add_gaussian_noise(std=n)
            for h2 in n2:
                print(f'Testing with noise={n}, h2={h2}')
                train_loss, test_loss, validation_loss = self.train_and_test(MLP(h2=h2), is_plot=False)
                losses.append(train_loss)
                test_losses.append(test_loss)
                order_refference.append(f'n:{n2} & noise:{noise}')
        plot_losses(losses, ' - Gaussian Noise', 'line ')
        graph_matrix(test_losses, noise, n2, 'Noise Standard Deviation', 'n2 (Number of Nodes in Second Layer)', 'Test Loss for Different Noise Levels (Noise vs n2)')
