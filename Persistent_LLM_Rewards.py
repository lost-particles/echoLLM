import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
from huggingface_hub import login  # Import the login function
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from datetime import datetime
import pickle
import argparse

print(torch.cuda.is_available())
# Log in to Hugging Face Hub
login_token = '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'
login(login_token)

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "Qwen/Qwen2.5-0.5B-Instruct" # 20 ep/min
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# model = AutoModelForCausalLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

env = gym.make("FrozenLake-v1", render_mode="ansi")
# Global dynamic conversation history (excluding static)
conversation_history_ids = None  # Tensor
#max_dynamic_tokens = None

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

def prepare_static_prompt(grid_map):
    static_prompt = (
        "### Instruction:\n"
        "You are evaluating a move made by an agent in the Frozen Lake game.\n"
        "The lake is a 4x4 grid with 16 states (0 to 15), where the agent starts at state 0 and must reach the goal at state 15.\n"
        "There are holes that will end the game if the agent falls in, and loops or unnecessary steps should be avoided.\n\n"
        f"The layout of the grid is: {grid_map}.\n\n"
        "When evaluating the move, consider the following:\n"
        "- Penalize moves that return the agent to a recently visited state without clear progress.\n"
        "- Reward moves that bring the agent closer to the goal or explore new parts of the map.\n"
        "- Repeated cycling between a small set of states should be discouraged.\n"
        "- Avoid rewarding safe but stagnant behavior (e.g., staying in a loop just to avoid holes).\n\n"
        "### History:\n"
    )
    return tokenizer(static_prompt, return_tensors="pt")["input_ids"].to(device)

def get_language_reward(
    state,
    action,
    next_state,
    static_input_ids,
    log_prompts=False,
):
    global conversation_history_ids
    global recent_history

    action_map = ['left', 'down', 'right', 'up']
    action_name = action_map[action]

    # Summarize the N most recent transitions
    window_summary = "\n".join(
        [f"{i + 1}. State {s} → [{a}] → State {ns}" for i, (s, a, ns) in enumerate(recent_history)]
    )

    # Current move text
    current_turn_text = (
        f"### Recent Transitions:\n{window_summary}\n\n"
        f"Agent: I am at state {state}.\n"
        f"Environment: You moved {action_name} to state {next_state}.\n\n"
        "How good was this move on a scale from 0 (very bad) to 1 (excellent)?\n"
        "Respond with a single decimal number only.\n"
        "### Response:\n"
    )
    current_turn_ids = tokenizer(current_turn_text, return_tensors="pt")["input_ids"].to(device)

    # Combine dynamic tokens
    if conversation_history_ids is None:
        dynamic_ids = current_turn_ids
    else:
        dynamic_ids = torch.cat([conversation_history_ids, current_turn_ids], dim=-1)

    # Trim dynamic history if needed
    if dynamic_ids.shape[-1] > max_dynamic_tokens:
        overflow = dynamic_ids.shape[-1] - max_dynamic_tokens
        dynamic_ids = dynamic_ids[:, overflow:]

    # Combine with static prompt (not counted for truncation)
    full_input_ids = torch.cat([static_input_ids, dynamic_ids], dim=-1)

    # Run the model
    outputs = model.generate(input_ids=full_input_ids, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if log_prompts:
        print("==== Prompt ====")
        print(tokenizer.decode(full_input_ids[0], skip_special_tokens=True))
        print("==== Response ====")
        print(response.strip())

    # Parse the response to get the reward
    try:
        reward_str = response.split("### Response:")[-1].strip()
        reward_val = float(reward_str.split()[0])
        reward_val = max(0.0, min(1.0, reward_val))
    except:
        reward_val = 0.0

    # Add full turn (prompt + response) to conversation history
    full_turn_text = current_turn_text + f"{reward_val}\n"
    full_turn_ids = tokenizer(full_turn_text, return_tensors="pt")["input_ids"].to(device)

    if conversation_history_ids is None:
        conversation_history_ids = full_turn_ids
    else:
        conversation_history_ids = torch.cat([conversation_history_ids, full_turn_ids], dim=-1)

    # Enforce dynamic history token limit after adding
    if conversation_history_ids.shape[-1] > max_dynamic_tokens:
        overflow = conversation_history_ids.shape[-1] - max_dynamic_tokens
        conversation_history_ids = conversation_history_ids[:, overflow:]

    return reward_val


# LLM Rewards

def q_learning_llm(env, num_episodes=5000, save_every=100, alpha=0.5, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    epsilon = initial_epsilon
    rewards_per_episode = []
    ep_num = 0
    env.reset()
    grid_map = env.render()
    global conversation_history_ids
    global sliding_summary_window
    global recent_history

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
        dump_files = [f for f in os.listdir(latest_run_dir) if f.endswith(".pkl") or f.endswith(".pt")]
        if dump_files:
            latest_dump_file = max(dump_files, key=lambda x: int(x.split('_')[1]))
            latest_dump_path = os.path.join(latest_run_dir, latest_dump_file)
            if latest_dump_path.endswith(".pkl"):
                with open(latest_dump_path, "rb") as f:
                    latest_data = pickle.load(f)
            elif latest_dump_path.endswith(".pt"):
                latest_data = torch.load(latest_dump_path, map_location=device, weights_only=False)

    if latest_data is not None:
        q_table = latest_data.get("q_table", q_table)
        ep_num = latest_data.get("ep_num", 0)
        epsilon = latest_data.get("epsilon", initial_epsilon)
        rewards_per_episode = latest_data.get("rewards_per_episode", [])
        conversation_history_ids = latest_data.get("conversation_history_ids", None)
        print(f"Loaded data from {latest_dump_path}")
    else:
        print("No previous data found. Starting fresh.")

    run_dir = os.path.join(model_root, timestamp)
    os.makedirs(run_dir, exist_ok=False)

    # One-time setup
    static_input_ids = prepare_static_prompt(grid_map)

    for i in range(ep_num, ep_num+num_episodes):
        ep_num = i + 1
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
            reward = get_language_reward(state,action,next_state,static_input_ids=static_input_ids,log_prompts=False)
            recent_history.append((state, action, next_state))
            if len(recent_history) > sliding_summary_window:
                recent_history.pop(0)
            steps += 1

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        total_reward /= steps
        rewards_per_episode.append(total_reward)
        if num_eps < 2000 and (i + 1) % 20 == 0:
            print(f"Episode {i+1} done")
            print(f'Summary of Recent History : {recent_history}')

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (i + 1) % save_every == 0:
            avg_reward = np.mean(rewards_per_episode[-save_every:])
            print(f"Episode {i + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            save_path = os.path.join(run_dir, f"ep_{i + 1}_stats.pt")
            save_data = {
                "q_table": q_table,
                "ep_num": ep_num,
                "epsilon": epsilon,
                "rewards_per_episode": rewards_per_episode,
                "conversation_history_ids": conversation_history_ids
            }
            torch.save(save_data, save_path)
    return q_table, rewards_per_episode


if __name__ == "__main__":
    global max_dynamic_tokens
    global sliding_summary_window
    global recent_history
    recent_history = []
    # required to provide number of episodes and save_every
    # provide the number of episodes and save_every as command line arguments
    parser = argparse.ArgumentParser(description="Run Q-Learning with LLM-based rewards.")
    parser.add_argument("--num_eps", type=int, default=5000, help="Number of episodes to run.")
    parser.add_argument("--save_every", type=int, default=100, help="Frequency of saving the model.")
    parser.add_argument("--max_dynamic_tokens", type=int, default=1024, help="Context length for the llm")
    parser.add_argument("--sliding_summary_window", type=int, default=20, help="Sliding window size for keeping previous moves, to summarize over")
    args = parser.parse_args()

    num_eps = args.num_eps
    save_every = args.save_every
    max_dynamic_tokens = args.max_dynamic_tokens # Only applies to dynamic tokens, static is always included
    print(f'using max_dynamic_tokens as {max_dynamic_tokens}')
    sliding_summary_window = args.sliding_summary_window
    start = datetime.now()
    print(f"Start time: {start}")
    q_table, rewards = q_learning_llm(env, num_episodes=num_eps, save_every=save_every)
    end = datetime.now()
    print(f"End time: {end}")
    print(f"Total time taken: {end - start}")

    # Plot the rewards
    plt.figure(figsize=(20, 10))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    # save the plot
    plot_path = "models"
    timestamps = [d for d in os.listdir(plot_path) if os.path.isdir(os.path.join(plot_path, d))]
    if timestamps:
        latest_timestamp = max(timestamps)
        plot_path = os.path.join(plot_path, latest_timestamp)
    else:
        plot_path = os.path.join(plot_path, "default_run")

    plot_path = os.path.join(plot_path, "rewards_plot.png")
    plt.savefig(plot_path)

    # Print the Q-Table
    print_q_table(q_table, env)

    """ DO NOT RUN ON OOD"""
    # env = gym.make("FrozenLake-v1", render_mode="rgb_array")
    # # Visualize the agent's performance
    # visualize_agent(env, q_table, episodes=1, sleep_time=0.5, end_sleep_time=1)

    # Clean up the environment
    env.close()