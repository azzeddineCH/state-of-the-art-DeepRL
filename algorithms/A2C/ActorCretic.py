import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCritic, self).__init__()

        self.action_size = action_size

        self.cretic_linear1 = nn.Linear(state_size, hidden_size)
        self.cretic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(state_size, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)

        value = F.relu(self.cretic_linear1(state))
        value = self.cretic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist