import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import grad


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


class SchrodingerPINN(nn.Module):
    def __init__(self, layers, lower, upper, activation=nn.Tanh()):
        super(SchrodingerPINN, self).__init__()
        self.layers = []
        self.lower = lower
        self.upper = upper
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(activation)
        self.model = nn.Sequential(*self.layers)
        self.model.apply(weights_init)  # xavier initialization
        assert self.layers[-1].out_features == 2  # real part and imaginary part
        assert self.lower.shape == self.upper.shape == torch.Size([self.layers[0].in_features])

    def forward(self, _time, *coordinates):
        assert self.layers[0].in_features == len([_time, *coordinates])
        # time and coordinates (x, y, z)
        psi = self.model((torch.cat([_time, *coordinates], dim=1) -
                          self.lower) / (self.upper - self.lower))  # Min-max scaling
        real = psi[:, 0:1]
        imaginary = psi[:, 1:2]
        return real, imaginary


class AllanChanPINN(nn.Module):
    def __init__(self, layers):
        super(AllanChanPINN, self).__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, t):
        u = self.model(torch.cat([x, t], dim=1))
        return u


class FisherKPPPINN(nn.Module):
    def __init__(self, layers):
        super(FisherKPPPINN, self).__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, t):
        u = self.model(torch.cat([x, t], dim=1))
        return u


class DiffusionPINN(nn.Module):
    def __init__(self, layers):
        super(DiffusionPINN, self).__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, t):
        u = self.model(torch.cat([x, t], dim=1))
        return u


class BurgerPINN(nn.Module):
    def __init__(self, layers):
        super(BurgerPINN, self).__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, t):
        u = self.model(torch.cat([x, t], dim=1))
        return u


class PoissonPINN(nn.Module):
    def __init__(self, layers, activation):
        super(PoissonPINN, self).__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(activation)
        self.model = nn.Sequential(*self.layers)

    def forward(self, input_data):
        u = self.model(input_data)
        return u