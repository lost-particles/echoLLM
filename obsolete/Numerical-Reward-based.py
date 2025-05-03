import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep

# Create the Frozen Lake environment
env = gym.make("FrozenLake-v1", render_mode="rgb_array")

def q_learning(env, num_episodes=5000, alpha=0.5, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    epsilon = initial_epsilon
    rewards_per_episode = []

    for i in range(num_episodes):
        state, _ = env.reset()  # Correct unpacking of the reset method
        done = False
        total_reward = 0

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, truncated, info = env.step(action)

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Print progress every 500 episodes
        if (i + 1) % 500 == 0:
            avg_reward = np.mean(rewards_per_episode[-500:])
            print(f"Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    return q_table, rewards_per_episode

# Train the agent
q_table, rewards = q_learning(env, num_episodes=20000, alpha=0.5, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995)

# Plot the rewards
plt.figure(figsize=(20, 10))
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Print the Q-Table
print_q_table(q_table, env)

# Visualize the agent's performance
visualize_agent(env, q_table, episodes=10, sleep_time=0.5, end_sleep_time=2)

# Clean up the environment
env.close()