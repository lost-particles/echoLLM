from huggingface_hub import login  # Import the login function
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os
from datetime import datetime
import pickle

# Log in to Hugging Face Hub
login_token = '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'
login(login_token)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

env = gym.make("FrozenLake-v1", render_mode="ansi")
num_eps = 5000

reward_cache = {}

def get_language_reward(state, action, next_state, grid_map):
    # key = (state, action, next_state)
    # if key in reward_cache:
    #     return reward_cache[key]

    action_map = ['left', 'down', 'right', 'up']
    action_name = action_map[action]

    prompt = (
        f"### Instruction:\n"
        f"You are evaluating a move made by an agent in the Frozen Lake game.\n"
        f"The lake is a 4x4 grid with 16 states (0 to 15), where the agent starts at state 0 and must reach the goal at state 15.\n"
        f"There are holes that will end the game if the agent falls in, and loops or unnecessary steps should be avoided.\n\n"
        f"The layout of the grid is: {grid_map}.\n"
        f"The agent moved from state {state} to state {next_state} by going {action_name}.\n"
        f"How good was this move on a scale from 0 (very bad) to 1 (excellent)?\n"
        f"Respond with a single decimal number only.\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device)


    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        reward_str = response.split("### Response:")[-1].strip()
        reward_val = float(reward_str.split()[0])
        reward_val = max(0.0, min(1.0, reward_val))
    except:
        reward_val = 0.0

    # reward_cache[key] = reward_val
    return reward_val

def q_learning_llm(env, num_episodes=5000, alpha=0.5, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    epsilon = initial_epsilon
    rewards_per_episode = []
    env.reset()
    grid_map = env.render()
    
    # Prepare model saving directories
    model_root = "models"
    os.makedirs(model_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(model_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    save_every = 500

    for i in range(num_episodes):
        state, _ = env.reset()
        # print(f"ENV: {env.render()}")
        done = False
        total_reward = 0
        steps = 0
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, _, done, truncated, info = env.step(action)

            # Replace native reward with LLM-generated reward
            reward = get_language_reward(state, action, next_state, grid_map)
            steps += 1

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        total_reward /= steps
        rewards_per_episode.append(total_reward)
        if (i + 1) % 20 == 0:
            print(f"Episode {i+1} done")

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (i + 1) % save_every == 0:
            avg_reward = np.mean(rewards_per_episode[-save_every:])
            print(f"Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            save_path = os.path.join(run_dir, f"q_table_ep{i+1}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(q_table, f)


    return q_table, rewards_per_episode

q_table, rewards = q_learning_llm(env, num_episodes=num_eps)
# Plot the rewards
plt.figure(figsize=(20, 10))
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


def visualize_agent(env, q_table, episodes=5, sleep_time=0.5, end_sleep_time=2):
    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            clear_output(wait=True)
            plt.imshow(env.render())
            plt.axis('off')
            plt.show()
            sleep(sleep_time)

            action = np.argmax(q_table[state])
            state, reward, done, truncated, info = env.step(action)

        clear_output(wait=True)
        plt.imshow(env.render())
        plt.axis('off')
        plt.show()
        sleep(end_sleep_time)

def print_q_table(q_table, env):
    """Prints the Q-table in a readable format using pandas DataFrame."""
    actions = ['Left', 'Down', 'Right', 'Up']
    df = pd.DataFrame(q_table, columns=actions)
    df.index.name = 'State'

    print("\n===== Q-Table =====")
    print(df.round(2))  # Round to 2 decimal places for readability
    print("===================\n")

# Print the Q-Table
print_q_table(q_table, env)

""" DO NOT RUN ON OOD"""
# env = gym.make("FrozenLake-v1", render_mode="rgb_array")
# # Visualize the agent's performance
# visualize_agent(env, q_table, episodes=1, sleep_time=0.5, end_sleep_time=1)

# Clean up the environment
env.close()
