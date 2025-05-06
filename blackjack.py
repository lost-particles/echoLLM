import os
import gymnasium as gym
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# ==== Setup ====
login_token = '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'
login(login_token)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

env = gym.make("Blackjack-v1", natural=False, sab=False)
action_map = ['stick', 'hit']

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

# ==== Prompting ====
def prepare_static_prompt_blackjack():
    examples = [
        "(13, 10, False) → hit → (23, 10, False) # Busted → 0.0",
        "(20, 10, False) → stick → (20, 10, False) # Safe stand → 1.0",
        "(12, 2, False) → stick → (12, 2, False) # Too passive → 0.3",
        "(16, 10, False) → hit → (18, 10, False) # Risk paid off → 0.8",
        "(18, 10, False) → hit → (24, 10, False) # Unnecessary risk → 0.1",
        "(19, 9, False) → stick → (19, 9, False) # Correct stand → 0.9",
    ]
    prompt = (
        "### Instruction:\n"
        "Evaluate Blackjack moves made by an agent. Return ONLY a score from 0.0 to 1.0.\n"
        "Format: (player_sum, dealer_card, usable_ace) → action → next_state\n\n"
        "Examples:\n" + "\n".join(examples) +
        "\n\n### Now evaluate:\n"
    )
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

def get_language_reward(state, action, next_state, static_prompt_ids, memory=None, log=False):
    action_name = action_map[action]
    history_str = ""
    if memory:
        history_str = "Recent:\n" + "\n".join(
            f"{s} → {action_map[a]} → {ns}" for s, a, ns in memory
        ) + "\n"

    prompt = (
        history_str +
        f"{state} → {action_name} → {next_state}\n"
        "Score (0.0 to 1.0): Respond ONLY with a single decimal number.\n### Response:\n"
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    full_input_ids = torch.cat([static_prompt_ids, prompt_ids], dim=-1)

    outputs = model.generate(full_input_ids, max_new_tokens=10)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if log:
        print("==== PROMPT ====\n", tokenizer.decode(full_input_ids[0], skip_special_tokens=True))
        print("==== RESPONSE ====\n", decoded.strip())

    for line in reversed(decoded.strip().splitlines()):
        try:
            score = float(line.strip())
            return np.clip(score, 0.0, 1.0)
        except:
            continue
    return 0.0

def summarize_experience(transitions):
    text = "Transitions:\n" + "\n".join(
        f"{s} → {ns} via {action_map[a]}" for s, a, ns in transitions
    )
    prompt = (
        "### Instruction:\nSummarize the agent’s Blackjack strategy.\n" +
        text + "\n### Summary:\n"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**input_ids, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==== Training ====
def q_learning_llm(env, num_episodes=400, memory_type="short"):
    obs_space = env.observation_space
    nS0, nS1, nS2 = obs_space[0].n, obs_space[1].n, obs_space[2].n
    nA = env.action_space.n
    q_table = np.zeros((nS0, nS1, nS2, nA))

    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.995
    alpha = 0.5
    gamma = 0.95

    rewards_per_episode = []
    win_draw_loss_log = []
    results = {"win": 0, "draw": 0, "loss": 0}

    memory = TransitionHistory(size=3)
    summary_text = ""
    static_prompt_ids = prepare_static_prompt_blackjack()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("models", f"blackjack_llm_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_llm_reward = 0
        memory.history.clear()
        episode_transitions = []

        while not done:
            s0, s1, s2 = state
            action = np.random.choice(nA) if np.random.rand() < epsilon else np.argmax(q_table[s0, s1, s2])
            next_state, env_reward, done, _, _ = env.step(action)

            memory.add(state, action, next_state)
            episode_transitions.append((state, action, next_state))

            if memory_type == "short":
                reward = get_language_reward(state, action, next_state, static_prompt_ids, memory.get(), log=False)
            elif memory_type == "summary":
                reward = get_language_reward(state, action, next_state, static_prompt_ids, memory=None, log=False)
            else:
                reward = get_language_reward(state, action, next_state, static_prompt_ids, memory=None, log=False)

            ns0, ns1, ns2 = next_state
            q_table[s0, s1, s2, action] += alpha * (reward + gamma * np.max(q_table[ns0, ns1, ns2]) - q_table[s0, s1, s2, action])
            state = next_state
            total_llm_reward += reward

        if memory_type == "summary" and (ep + 1) % 100 == 0:
            summary_text = summarize_experience(episode_transitions)

        if env_reward == 1:
            results["win"] += 1
            outcome = "win"
        elif env_reward == 0:
            results["draw"] += 1
            outcome = "draw"
        else:
            results["loss"] += 1
            outcome = "loss"

        rewards_per_episode.append(total_llm_reward)
        win_draw_loss_log.append(outcome)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            print(f"[{ep+1}/{num_episodes}] Avg reward: {avg:.3f} | Epsilon: {epsilon:.3f}")
            print(f"→ Wins: {results['win']} | Draws: {results['draw']} | Losses: {results['loss']}")
            with open(os.path.join(run_dir, f"q_table_ep{ep+1}.pkl"), "wb") as f:
                pickle.dump(q_table, f)

    with open(os.path.join(run_dir, "final_q_table.pkl"), "wb") as f:
        pickle.dump(q_table, f)
    with open(os.path.join(run_dir, "rewards.pkl"), "wb") as f:
        pickle.dump(rewards_per_episode, f)
    with open(os.path.join(run_dir, "results_log.pkl"), "wb") as f:
        pickle.dump(win_draw_loss_log, f)

    print("\n✅ Training Complete.")
    print(f"Final → Wins: {results['win']} | Draws: {results['draw']} | Losses: {results['loss']}")

    return q_table, rewards_per_episode, run_dir



def play_blackjack_games(q_table_path, num_episodes=10, show_llm_scores=True, max_steps=20):
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)

    static_ids = prepare_static_prompt_blackjack()

    results = {"win": 0, "draw": 0, "loss": 0}

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        memory = TransitionHistory(3)
        steps = 0

        print(f"\n🎲 Episode {ep+1}/{num_episodes}")

        while not done and steps < max_steps:
            s0, s1, s2 = state
            action = np.argmax(q_table[s0, s1, s2])
            next_state, env_reward, done, _, info = env.step(action)
            action_name = action_map[action]

            if show_llm_scores:
                score = get_language_reward(state, action, next_state, static_ids, memory.get(), log=True)
                print(f"{steps+1:02d}. {state} → {action_name.upper()} → {next_state} | LLM score: {score:.2f}")
            else:
                print(f"{steps+1:02d}. {state} → {action_name.upper()} → {next_state}")

            memory.add(state, action, next_state)
            state = next_state
            steps += 1

        # Determine result
        if env_reward == 1:
            print("✅ Agent WON the game!")
            results["win"] += 1
        elif env_reward == 0:
            print("➖ Agent DREW the game.")
            results["draw"] += 1
        else:
            print("❌ Agent LOST the game.")
            results["loss"] += 1

    # Summary
    print("\n📊 Summary after", num_episodes, "games:")
    print(f"✅ Wins:  {results['win']}")
    print(f"➖ Draws: {results['draw']}")
    print(f"❌ Losses:{results['loss']}")



if __name__ == "__main__":

    # ==== Run Training ====
    q_table, rewards, save_dir = q_learning_llm(env, num_episodes=400, memory_type="short")

    # ==== Reward Plot ====
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.title("LLM Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total LLM Reward")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "rewards_plot.png"))
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(rewards, bins=20)
    plt.title("LLM Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "reward_histogram.png"))
    plt.show()





