from algorithms.A2C.ActorCretic import ActorCritic
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys



# hyperparameters
hidden_size = 256
learning_rate = 3e-4


# Constants
GAMMA = 0.99
nb_steps = 300
nb_episodes = 3000


class Agent():

    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.actor_critic = ActorCritic(
            self.state_size, self.action_size, hidden_size)

    def run(self):
        self.actor_critic.train()

        ac_optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=learning_rate)

        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0

        for episode in range(nb_episodes):

            rewards = []
            values = []
            log_probs = []

            state = self.env.reset()

            for step in range(nb_steps):

                value, policy_dist = self.actor_critic.forward(state)
                value = value.detach().numpy()[0, 0]
                dist = policy_dist.detach().numpy()

                action = np.random.choice(self.action_size, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))

                new_state, reward, done, _ = self.env.step(action)

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)

                entropy_term += entropy

                state = new_state

                if done or step == nb_steps-1:
                    QVal, _ = self.actor_critic.forward(new_state)
                    QVal = QVal.detach().numpy()[0, 0]
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(step)
                    average_lengths.append(np.mean(all_lengths[-10:]))
                    if episode % 10 == 0:
                        sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(
                            episode, np.mean(all_rewards), nb_steps, average_lengths[-1]))
                    break

            # Compute Q values
            QVals = np.zeros_like(values)

            for r in reversed(range(len(values))):
                QVal = rewards[r] + GAMMA * QVal
                QVals[r] = QVal

            # update actor critic

            values = torch.FloatTensor(values)
            QVals = torch.FloatTensor(QVals)

            log_probs = torch.stack(log_probs, dim=-1)

            advantage = QVals - values

            actor_loss = -(log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()

            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

            ac_optimizer.zero_grad()
            ac_loss.backward()
            ac_optimizer.step()

        # Plot results
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

    def act(self, state):

        self.actor_critic.eval()

        _, policy_dist = self.actor_critic.forward(state)
        policy_dist = policy_dist.detach().numpy()

        action = np.random.choice(self.action_size, p=np.squeeze(policy_dist))

        return action
