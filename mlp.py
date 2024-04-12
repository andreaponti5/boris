import numpy as np
import torch

from scipy.stats._qmc import LatinHypercube
from sklearn.neural_network import MLPRegressor
from torch import nn

from objectives import alpine01


def regressor(x, y, hidden_layers_size):
    return MLPRegressor(hidden_layer_sizes=[hidden_layers_size] * 3,
                        max_iter=20000,
                        random_state=1).fit(x, y)


def predict(model, x, bounds):
    y = model.predict(x)
    y[y < bounds[0]] = bounds[0]
    y[y > bounds[1]] = bounds[1]
    return torch.Tensor(y)


def mlp_torch(x, y, hidden_layers_size):
    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden_layers_size),
        nn.ReLU(),
        nn.Linear(hidden_layers_size, hidden_layers_size),
        nn.Dropout(0.7),
        nn.ReLU(),
        nn.Linear(hidden_layers_size, hidden_layers_size),
        nn.Dropout(0.7),
        nn.ReLU(),
        nn.Linear(hidden_layers_size, hidden_layers_size),
        nn.ReLU(),
        nn.Linear(hidden_layers_size, y.shape[1])
    )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []
    pred_y = None
    for epoch in range(2000):
        pred_y = model(x)
        loss = loss_function(pred_y, y)
        print(f"Epoch {epoch}\tloss = {loss:.6f}")
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()

        optimizer.step()
    return model
