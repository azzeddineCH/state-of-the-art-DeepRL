import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from algorithms.DoubleDQN_PER.Network import Network as QNetwork
from algorithms.DoubleDQN_PER.PrioritizedReplayBuffer import PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 25
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 10
hidden_size = 256


class Agent():

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(0)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, hidden_size)
        self.qnetwork_target = QNetwork(
            state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, 0)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, eps):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, importance, indices = self.memory.sample(priority_scale=0.7)
                self.learn(experiences, importance, indices, GAMMA, eps)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, importance, indices, gamma, eps):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        selected_actions = self.qnetwork_local(next_states).detach().max(1)[1]

        Q_targets_next = self.qnetwork_target(next_states).detach()

        Q_targets_next = Q_targets_next[np.arange(
            len(Q_targets_next)), selected_actions].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        errors = Q_expected - Q_targets

        self.memory.set_priorities(indices, errors.detach().numpy().squeeze(1))

        importance = torch.from_numpy(importance).float()
        # Compute loss
        loss = torch.mean(torch.mul(errors ** 2, importance**(1-eps)))
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)
