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
import argparse
import json

# Log in to Hugging Face Hub
login_token = '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'
login(login_token)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

env = gym.make("FrozenLake-v1", render_mode="ansi")

reward_cache = {}

def get_language_reward(state, action, next_state, grid_map):
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

def q_learning_llm(env, num_episodes=5000, save_every=100, alpha=0.5, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    epsilon = initial_epsilon
    rewards_per_episode = []
    ep_num = 0
    env.reset()
    grid_map = env.render()
    
    # Prepare model saving directories
    model_root = "models"

    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    except Exception as e:
        print(f"Error generating timestamp: {e}")
        timestamp = "default_timestamp"
        
    # Load the most recent q_table, ep_num, epsilon, and rewards_per_episode if they exist
    latest_data = None
    timestamps = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
    if timestamps:
        latest_timestamp = max(timestamps)
        latest_run_dir = os.path.join(model_root, latest_timestamp)
        json_files = [f for f in os.listdir(latest_run_dir) if f.endswith(".json")]
        if json_files:
            latest_json_file = max(json_files, key=lambda x: int(x.split('_')[1]))
            latest_json_path = os.path.join(latest_run_dir, latest_json_file)
            with open(latest_json_path, "r") as f:
                latest_data = json.load(f)

    if latest_data is not None:
        q_table = np.array(latest_data.get("q_table", q_table))
        ep_num = latest_data.get("ep_num", 0)
        epsilon = latest_data.get("epsilon", initial_epsilon)
        rewards_per_episode = latest_data.get("rewards_per_episode", [])
        print(f"Loaded data from {latest_json_path}")
    else:
        print("No previous data found. Starting fresh.")
    run_dir = os.path.join(model_root, timestamp)
    os.makedirs(run_dir, exist_ok=False)

    for i in range(ep_num, num_episodes):
        ep_num = i+1
        state, _ = env.reset()
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
            save_path = os.path.join(run_dir, f"ep_{i+1}_stats.json")
            save_data = {
                "q_table": q_table,
                "ep_num": ep_num,
                "epsilon": epsilon,
                "rewards_per_episode": rewards_per_episode
            }
            with open(save_path, "w") as f:
                json.dump(save_data, f)

    return q_table, rewards_per_episode

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

if __name__ == "__main__":
    # required to provide number of episodes and save_every
    # provide the number of episodes and save_every as command line arguments
    parser = argparse.ArgumentParser(description="Run Q-Learning with LLM rewards on FrozenLake.")
    parser.add_argument("--num_eps", type=int, default=1000, help="Number of episodes to run.")
    parser.add_argument("--save_every", type=int, default=100, help="Frequency of saving the model.")
    args = parser.parse_args()

    num_eps = args.num_eps
    save_every = args.save_every
    q_table, rewards = q_learning_llm(env, num_episodes=num_eps, save_every=save_every)
    # Plot the rewards
    plt.figure(figsize=(20, 10))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # Print the Q-Table
    print_q_table(q_table, env)

    """ DO NOT RUN ON OOD"""
    # env = gym.make("FrozenLake-v1", render_mode="rgb_array")
    # # Visualize the agent's performance
    # visualize_agent(env, q_table, episodes=1, sleep_time=0.5, end_sleep_time=1)

    # Clean up the environment
    env.close()
