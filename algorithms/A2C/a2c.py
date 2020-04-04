import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

from algorithms.A2C.ActorCretic import ActorCritic

nb_episodes = 1000
nb_steps = 100

# hyperparameters
hidden_size = 256
learning_rate = 3e-4


# Constants
GAMMA = 0.99
nb_steps = 300
nb_episodes = 300


def A2C(env):

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor_critic = ActorCritic(state_size, action_size, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(nb_episodes):

        rewards = []
        values = []
        log_probs = []

        state = env.reset()

        for step in range(nb_steps):

            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(action_size, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            entropy_term += entropy

            state = new_state

            if done or step == nb_steps-1:
                QVal, _ = actor_critic.forward(new_state)
                QVal = QVal.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(step)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(
                        episode, np.sum(rewards), nb_steps, average_lengths[-1]))
                break

        # Compute Q values

        QVals = np.zeros_like(values)

        for r in reversed(range(len(values))):
            QVal = rewards[r] + GAMMA * QVal
            QVals[r] = QVal

        # update actor critic

        values = torch.FloatTensor(values)
        QVals = torch.FloatTensor(QVals)
        log_probs = torch.stack(log_probs)

        advantage = QVals - values

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
