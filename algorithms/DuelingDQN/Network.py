import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, state_size, action_size,hidden_size):
        super(Network, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.state_value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):

        features = self.feature_layer(x)

        adavantages = self.advantage_layer(features)
        state_value = self.state_value_layer(features)

        output = state_value + (adavantages - adavantages.mean())

        return output
