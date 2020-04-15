from collections import deque, namedtuple
import numpy as np
import torch
import random


class PrioritizedReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / np.sum(scaled_priorities)
        return sample_probabilities

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def get_importance(self, probabilities):
        importance = (1 / self.batch_size) * (1/probabilities)
        importance_normalized = importance/max(importance)

        return importance_normalized

    def sample(self, priority_scale=1.0):
        """Randomly sample a batch of experiences from memory."""
        sample_probs = self.get_probabilities(priority_scale)

        sample_indices = random.choices(
            range(len(self.priorities)), k=self.batch_size, weights=sample_probs)
        importance = self.get_importance(sample_probs[sample_indices])

        experiences = []
        for i in sample_indices:
            experiences.append(self.memory[i])

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones), importance, sample_indices

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
