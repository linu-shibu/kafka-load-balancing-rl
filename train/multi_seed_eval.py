import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from env.kafka_env import KafkaLoadBalancingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SEEDS = [0, 1, 2, 3, 4]
NUM_EPISODES_PER_SEED = 20
NUM_BROKERS = 4
EPISODE_LENGTH = 100

def run_episodes_seeded(env, policy_fn, seed, model=None):
    env.seed(seed)
    episode_rewards = []
    for _ in range(NUM_EPISODES_PER_SEED):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = [policy_fn(env, obs)]
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
        episode_rewards.append(total_reward)
    return episode_rewards

def round_robin_policy(env, obs):
    return env.envs[0].current_step % NUM_BROKERS

def least_loaded_policy(env, obs):
    broker_loads = env.envs[0].broker_loads
    return int(np.argmin(broker_loads))

if __name__ == "__main__":
    model_env = KafkaLoadBalancingEnv(num_brokers=NUM_BROKERS, episode_length=EPISODE_LENGTH)
    model_env = DummyVecEnv([lambda: model_env])
    model_env = VecNormalize.load("../models/vecnormalize_v3.pk1", model_env)
    model_env.training = False
    model_env.norm_reward = False
    model = PPO.load("../models/kafka_ppo_v3")

    heuristic_env = DummyVecEnv([lambda: KafkaLoadBalancingEnv(num_brokers=NUM_BROKERS, episode_length=EPISODE_LENGTH)])

    all_results = {"PPO": [], "Least-loaded": [], "Round-robin": []}

    for seed in SEEDS:
        all_results["PPO"].extend(run_episodes_seeded(model_env, None, seed, model=model))
        all_results["Least-loaded"].extend(run_episodes_seeded(heuristic_env, least_loaded_policy, seed))
        all_results["Round-robin"].extend(run_episodes_seeded(heuristic_env, round_robin_policy, seed))
    
    print(f"\n{'Policy':<15} {'Mean':>10} {'Std Dev':>10} {'Min':>10} {'Max':>10}")
    print("-" * 57)
    for name, rewards in all_results.items():
        print(f"{name:<15} {np.mean(rewards):>10.2f} {np.std(rewards):>10.2f} {np.min(rewards):>10.2f} {np.max(rewards):>10.2f}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [all_results["PPO"],all_results["Least-loaded"], all_results["Round-robin"]],
        tick_labels=["PPO", "Least-loaded", "Round-robin"]
    )
    ax.set_ylabel("Episode Reward (higher/less negative = better)")
    ax.set_title(f"Policy Comparison Across {len(SEEDS)} Seeds ({NUM_EPISODES_PER_SEED} episodes each)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results_plot.png", dpi=150)
    print("\nPlot saved to results_plot.png")