import gym
import matplotlib.pyplot as plt
import numpy as np

from a2c import A2CAgent

env = gym.make("CartPole-v0")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
MAX_EPISODE = 1000
MAX_STEPS = 500

lr = 7e-3
gamma = 0.99
value_coeff = 0.5
entropy_coeff = 1e-4

agent = A2CAgent(env, gamma, lr, value_coeff, entropy_coeff)

ep_rewards = []
for episode in range(MAX_EPISODE):
    state = env.reset()
    trajectory = []  # [[s, a, r, s', done], [], ...]
    episode_reward = 0
    for steps in range(MAX_STEPS):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        trajectory.append([state, action, reward, next_state, done])
        episode_reward += reward

        if done:
            break

        state = next_state
    ep_rewards.append(episode_reward)
    if episode % 20 == 0:
        print("Episode " + str(episode) + ": " + str(episode_reward))
    agent.update(trajectory)


plt.style.use('seaborn')
plt.plot(np.arange(0, len(ep_rewards), 5), ep_rewards[::5])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.savefig('plot.png')
