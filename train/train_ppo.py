import sys
sys.path.append('..')
from env.kafka_env import KafkaLoadBalancingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = KafkaLoadBalancingEnv(num_brokers=4, episode_length=100)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

model.learn(total_timesteps=300_000)

model.save("../models/kafka_ppo_v3")
env.save("../models/vecnormalize_v3.pkl")
print("Training complete. Model saved to models/kafka_ppo_v3")