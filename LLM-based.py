from huggingface_hub import login  # Import the login function

# Log in to Hugging Face Hub
login_token = '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'
login(login_token)

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, T5ForConditionalGeneration
import torch

# Load the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name = "meta-llama/Llama-3.2-3B-Instruct" # smaller
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# model_name = "google/flan-t5-xl"  # Using Flan-T5-XL
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_language_reward(state, action, next_state):
    action_map = ['left', 'down', 'right', 'up']
    action_name = action_map[action]

    prompt = (
        f"### Instruction:\n"
        f"In the Frozen Lake game, the agent moved from state {state} to state {next_state} by going {action_name}. "
        f"The goal is to reach state 15. Avoid holes and unnecessary loops. "
        f"How good was this move on a scale from 0 to 1? "
        f"Only return a number and in the following format:\n"
        f"0.2f\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the numeric response
    try:
        reward_str = response.split("### Response:")[-1].strip()
        print(f"RAW REWARD: ***{response}***\n")
        reward_val = float(reward_str.split()[0])
        # print(f"RAW reward number: ***{reward_val}***\n")
        return max(0.0, min(1.0, reward_val))
    except Exception as e:
        print(f"Error parsing reward: {e}")
        return 0.0
    
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep

def q_learning_llm(env, num_episodes=5000, alpha=0.5, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    epsilon = initial_epsilon
    rewards_per_episode = []

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, _, done, truncated, info = env.step(action)

            # Replace native reward with LLM-generated reward
            reward = get_language_reward(state, action, next_state)

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        if num_eps < 100:
            print(f"Episode {i+1} done")

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (i + 1) % 500 == 0:
            avg_reward = np.mean(rewards_per_episode[-500:])
            print(f"Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    return q_table, rewards_per_episode

# Visualization, Q-table printout, and other functions remain unchanged
# -- omitted here for brevity but you'd use the same ones as in your original code --

# Use the new LLM-based Q-learning function
env = gym.make("FrozenLake-v1", render_mode="rgb_array")
num_eps = 1000
q_table, rewards = q_learning_llm(env, num_episodes=num_eps)


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