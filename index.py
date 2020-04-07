import matplotlib.pyplot as plt
import gym
from algorithms.A2C.Agent import Agent as A2CAgent
from algorithms.DoubleDQL.DQN import dqn


def test(agent, episodes):

    scores = []
    for i in range(episodes):

        state = env.reset()
        done = False
        score = 0
        while done is False:

            action = agent.act(state)

            new_state, reward, done, _ = env.step(action)
            score += reward

            state = new_state

        scores.append(score)

    plot_res(scores)


def plot_res(scores):

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax[0].plot(scores, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(scores))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, scores, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(scores[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    ddql_agent = dqn(env)

    test(ddql_agent, 200)
