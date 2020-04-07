import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        output = F.relu(self.fc1(state))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output
