import unittest

import torch
from torch import nn, optim

from Lab01b.part1 import plot_loss, plot_losses
from utils import MackeyGlass
from Lab01b.graphing_matrix import graph_matrix

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
        loss = criterion(output, train_Y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        # 验证集上评估
        model.eval()
        with torch.no_grad():
            val_output = model(val_X)
            val_loss = criterion(val_output, val_Y.unsqueeze(1))

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


def test_model(model, test_X, test_Y):
    model.eval()
    with torch.no_grad():
        predictions = model(test_X).reshape(-1)
    accuracy = 1 - torch.mean(torch.abs(predictions - test_Y))
    print(f'Accuracy: {accuracy}')
    return accuracy.item()


class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        X, Y = MackeyGlass().generate_data()
        # average device the data into training, validation and testing
        self.X_train = X[:400]
        self.Y_train = Y[:400]
        self.X_val = X[401:800]
        self.Y_val = Y[401:800]
        self.X_test = X[800:]
        self.Y_test = Y[800:]

    def train_and_test(self, model: MLP = None, regularization=0.001, epochs=30000, study_rate=0.001, is_plot=True):
        if model is None:
            model = MLP()
        losses = train(model, torch.tensor(self.X_train).float(), torch.tensor(self.Y_train).float(),
                       torch.tensor(self.X_val).float(), torch.tensor(self.Y_val).float(), epochs, study_rate,
                       regularization, msg=False)
        if is_plot:
            plot_loss(losses)
        accuracy = test_model(model, torch.tensor(self.X_test).float(), torch.tensor(self.Y_test).float())
        return losses, accuracy

    def test(self):
        self.train_and_test()

    def test_dim_cases(self):
        losses = []
        accuracies = []
        n1 = [3, 4, 5]
        n2 = [2, 4, 6]
        for h1 in n1:
            for h2 in n2:
                print(f'Testing with h1={h1}, h2={h2}')
                loss, accuracy = self.train_and_test(MLP(h1, h2), is_plot=False)
                losses.append(loss)
                accuracies.append(accuracy)
        plot_losses(losses, ' - Hidden Dimension', 'line ')
        graph_matrix(accuracies, n1, n2)