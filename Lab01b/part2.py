import unittest

import torch
from torch import nn, optim

from DataGenerator import MackeyGlass


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, 1)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.Sigmoid(self.fc3(x))
        x = self.Sigmoid(self.fc4(x))
        return x


def train(model: nn.Module, train_X, train_Y, val_X, val_Y, epochs=3000, study_rate=0.001, regularization=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=study_rate, weight_decay=regularization if regularization else 0.0)

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

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')


def test_model(model, test_X):
    model.eval()
    with torch.no_grad():
        predictions = model(test_X).numpy()
    return predictions


class Test(unittest.TestCase):
    def test(self):
        X, Y = MackeyGlass().generate_data()
        # average device the data into training, validation and testing
        X_train = X[:400]
        Y_train = Y[:400]
        X_val = X[401:800]
        Y_val = Y[401:800]
        X_test = X[800:]
        Y_test = Y[800:]
        mlp = MLP()
        train(mlp, torch.tensor(X_train).float(), torch.tensor(Y_train).float(), torch.tensor(X_val).float(),
              torch.tensor(Y_val).float(), regularization=0.001)
        predictions = test_model(mlp, torch.tensor(X_test).float())
        print(predictions)
