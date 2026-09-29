import sys
sys.path.append('..')
import numpy as np
from env.kafka_env import KafkaLoadBalancingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

NUM_EPISODES = 50
NUM_BROKERS = 4
EPISODE_LENGTH = 100

def round_robin_policy(env, obs):
    return env.envs[0].current_step % NUM_BROKERS

def least_loaded_policy(env, obs):
    broker_loads = env.envs[0].broker_loads
    return int(np.argmin(broker_loads))

def random_policy(env, obs):
    return env.action_space.sample()

def run_episodes(env, policy_fn):
    episode_rewards = []
    for _ in range(NUM_EPISODES):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = [policy_fn(env, obs)]
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards), np.std(episode_rewards)

if __name__ == "__main__":
    heuristic_env = DummyVecEnv([lambda: KafkaLoadBalancingEnv(num_brokers=NUM_BROKERS, episode_length=EPISODE_LENGTH)])

    results = {}
    results["Random"] = run_episodes(heuristic_env, random_policy)
    results["Round-robin"] = run_episodes(heuristic_env, round_robin_policy)
    results["Least-loaded"] = run_episodes(heuristic_env, least_loaded_policy)

    print(f"\n{'Policy':<20} {'Mean Reward':>15} {'Std Dev':>10}")
    print("-" * 47)
    for name, (mean, std) in results.items():
        print(f"{name:<20} {mean:>15.2f} {std:>10.2f}")