import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Load LLM model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Blackjack LLM-based reward function
def get_language_reward_blackjack(player_sum, dealer_card, usable_ace, action):
    prompt = (
        f"### Instruction:\n"
        f"You are evaluating a move made by an agent in the Blackjack game.\n"
        f"The agent wants to beat the dealer by having a hand value closer to 21, without going over.\n"
        f"Cards are drawn from an infinite deck with replacement. Face cards (J, Q, K) are worth 10, and aces can count as 11 (usable ace) or 1.\n"
        f"The dealer shows one card and draws until reaching 17 or more.\n"
        f"Rewards are:\n"
        f"- Win: +1\n- Lose: -1\n- Draw: 0\n- Win with natural blackjack: +1.5 (natural=True), +1 (natural=False)\n\n"
        f"Game state:\n"
        f"- Player's hand total: {player_sum}\n"
        f"- Dealer's visible card: {dealer_card}\n"
        f"- Usable ace in hand: {'Yes' if usable_ace else 'No'}\n"
        f"- The agent chose to: {'hit' if action == 1 else 'stick'}\n\n"
        f"Evaluate this decision from 0 (very poor) to 1 (excellent), based on the agent's chance of winning or avoiding a bust.\n"
        f"Respond with a single decimal number only.\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        reward_str = response.split("### Response:")[-1].strip()
        reward_val = float(reward_str.split()[0])
        reward_val = max(0.0, min(1.0, reward_val))
    except:
        reward_val = 0.0

    return reward_val

# Q-learning for Blackjack
def q_learning_llm_blackjack(env, num_episodes=10000, alpha=0.5, gamma=0.95, initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):
    q_table = {}
    epsilon = initial_epsilon
    rewards_per_episode = []

    # Prepare model saving directories
    model_root = "models"
    os.makedirs(model_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(model_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    save_every = 40

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = tuple(obs)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_obs, native_reward, done, truncated, info = env.step(action)
            next_state = tuple(next_obs)
            steps += 1

            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)

            player_sum, dealer_card, usable_ace = state
            reward = get_language_reward_blackjack(player_sum, dealer_card, usable_ace, action)

            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            state = next_state
            total_reward += reward

        total_reward /= steps
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1} done")

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % save_every == 0:
            avg_reward = np.mean(rewards_per_episode[-save_every:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            save_path = os.path.join(run_dir, f"q_table_ep{episode+1}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(q_table, f)

    return q_table, rewards_per_episode

# Run training
env = gym.make("Blackjack-v1")
q_table, rewards = q_learning_llm_blackjack(env, num_episodes=500)

# Plot results
plt.plot(rewards)
plt.title("Rewards per Episode - Blackjack with LLM")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
