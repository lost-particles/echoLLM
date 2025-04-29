from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep
import os
from datetime import datetime
import pickle

login_token = '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'
login(login_token)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

env = gym.make("Blackjack-v1", natural=False, sab=False)
# Observation: (player_sum:4–31, dealer_card:1–10, usable_ace:0/1)
# Actions: 0=Stick, 1=Hit

class TransitionHistory:
    def __init__(self, size=5):
        self.history = []
        self.size = size

    def add(self, state, action, next_state):
        self.history.append((state, action, next_state))
        if len(self.history) > self.size:
            self.history.pop(0)

    def get(self):
        return self.history.copy()

def summarize_experience(episode_history):
    action_map = ['stick', 'hit']
    summary_prompt = (
        "### Instruction:\n"
        "Summarize this agent’s experience in Blackjack. What mistakes or good moves did it make?\n"
        "Transitions:\n" +
        "\n".join([
            f"{s} -> {ns} via {action_map[a]}"
            for s, a, ns in episode_history
        ]) +
        "\n### Response:\n"
    )
    inputs = tokenizer(summary_prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def get_language_reward(state, action, next_state, memory=None, summary_text=None):
    action_map = ['stick', 'hit']
    action_name = action_map[action]

    history_str = ""
    if memory:
        history_str = "Recent transitions:\n" + "\n".join(
            [f"{h[0]} -> {h[2]} via {action_map[h[1]]}" for h in memory]
        ) + "\n"

    summary_str = f"\nEpisode summary: {summary_text}\n" if summary_text else ""

    prompt = (
        "### Instruction:\n"
        "You are evaluating a move made by an agent in the Blackjack game.\n"
        "The agent wants to beat the dealer by having a hand value closer to 21, without going over.\n"
        "Cards are drawn from an infinite deck with replacement. Face cards (J, Q, K) are worth 10, and aces can count as 11 (usable ace) or 1.\n"
        "The dealer shows one card and draws until reaching 17 or more.\n"
        "State format: (player_sum, dealer_showing, usable_ace).\n"
        f"{history_str}"
        f"Move: {state} -> {next_state} via {action_name}."
        f"{summary_str}"
        "\nRate this move from 0 (bad) to 1 (excellent).\n"
        "Respond with only a decimal number (e.g. 0.23).\n"
        "### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        score = float(resp.strip().split("\n")[-3])
        return max(0.0, min(1.0, score))
    except:
        return 0.0
    
def q_learning_llm(env, num_episodes, memory_type="none",
                   alpha=0.5, gamma=0.95,
                   initial_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995):

    # Q-table: [player_sum (0–31), dealer_card (0–10), usable_ace (0–1), action (0–1)]
    obs_space = env.observation_space
    nS0, nS1, nS2 = obs_space[0].n, obs_space[1].n, obs_space[2].n
    nA = env.action_space.n
    q_table = np.zeros((nS0, nS1, nS2, nA))

    epsilon = initial_epsilon
    rewards_per_episode = []

    model_root = "models"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(model_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    save_every = 100
    memory = TransitionHistory(size=3)
    summary_memory = ""
    full_history = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        memory.history.clear()
        full_history.clear()

        while not done:
            # ε-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                s0, s1, s2 = state
                action = np.argmax(q_table[s0, s1, s2])

            # next_state, _, terminated, truncated, _ = env.step(action)
            # done = terminated or truncated
            next_state, env_reward, done, truncated, info = env.step(action)


            # record memory
            memory.add(state, action, next_state)
            full_history.append((state, action, next_state))

            # shaped reward
            # if memory_type == "short":
            #     r = get_language_reward(state, action, next_state, memory=memory.get())
            # elif memory_type == "summary":
            #     r = get_language_reward(state, action, next_state, summary_text=summary_memory)
            # else:
            #     r = get_language_reward(state, action, next_state)

            if memory_type=="none":
                r = env_reward
            else:
                r = get_language_reward(state, action, next_state, 
                                        memory=memory.get() if memory_type=="short" else None,
                                        summary_text=summary_memory if memory_type=="summary" else None)
            # Q-update
            s0, s1, s2 = state
            ns0, ns1, ns2 = next_state
            best_next = np.max(q_table[ns0, ns1, ns2])
            q_table[s0, s1, s2, action] += alpha * (r + gamma * best_next - q_table[s0, s1, s2, action])

            state = next_state
            total_reward += r
            steps += 1

        rewards_per_episode.append(total_reward / max(1, steps))

 
        if memory_type=="summary" and (ep+1) % 100 == 0:
            summary_memory = summarize_experience(full_history)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # checkpoint Q-table
        if (ep+1) % save_every == 0:
            path = os.path.join(run_dir, f"q_table_ep{ep+1}.pkl")
            with open(path, "wb") as f:
                pickle.dump(q_table, f)
            print(f"[{ep+1}/{num_episodes}] saved Q-table; ε={epsilon:.3f}")

    return q_table, rewards_per_episode

# --- Run & Plot ---
memory_type = "short"  # or "none", "summary"
q_table, rewards = q_learning_llm(env, num_episodes=500, memory_type=memory_type)

plt.figure(figsize=(12, 6))
plt.plot(rewards)
plt.title(f"Rewards per Episode ({memory_type})")
plt.xlabel("Episode")
plt.ylabel("Avg shaped reward")
plt.show()

