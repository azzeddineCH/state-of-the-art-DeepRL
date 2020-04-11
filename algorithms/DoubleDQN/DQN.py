
from collections import deque, namedtuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from algorithms.DoubleDQN.Agent import Agent


def dqn(env, n_episodes=5000, eps_start=1.0, eps_end=0.001, eps_decay=0.995):

    scores = []
    average_scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    env = env
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)

    for i_episode in range(n_episodes):

        state = env.reset()
        score = 0
        done = False

        while done == False:

            action = agent.act(np.asarray(state), eps)

            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

        scores_window.append(score)
        scores.append(score)

        eps = max(eps_end, eps_decay * eps)

        if (i_episode + 1) % 100 == 0:

            average_scores.append(np.mean(scores_window))
            print(f'Episode {i_episode + 1}, average {np.mean(scores_window)}')
        if (np.mean(scores_window) > 190):
            break

    # Plot results
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    return agent
