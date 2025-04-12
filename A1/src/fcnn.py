import torch.nn as nn


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(FullyConnectedNN, self).__init__()

        self.flatten = nn.Flatten()

        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.linear_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_layers(x)
        return out
