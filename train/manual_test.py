import sys
sys.path.append('..')
from env.kafka_env import KafkaLoadBalancingEnv

env = KafkaLoadBalancingEnv(num_brokers=4, episode_length=10)
obs, info = env.reset()
print("Initial observation: ", obs)

total_reward = 0
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"Step {step}: action={action}, reward={reward:.3f}, obs={obs}")
    if terminated or truncated:
        print("Episode ended")
        break

print(f"\nTotal reward: {total_reward:.3f}")