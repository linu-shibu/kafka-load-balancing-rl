import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KafkaLoadBalancingEnv(gym.Env):
    def __init__(self, num_brokers=4, episode_length=100):
        super().__init__()
        self.num_brokers = num_brokers
        self.episode_length = episode_length

        self.action_space = spaces.Discrete(num_brokers)

        obs_dim = num_brokers + num_brokers + num_brokers + num_brokers + 1
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.broker_loads = np.zeros(self.num_brokers, dtype=np.float32)
        self.broker_partition_counts = np.zeros(self.num_brokers, dtype=np.float32)

        self.broker_baseline = np.full(self.num_brokers, 5.0, dtype=np.float32)

        self.spike_remaining = np.zeros(self.num_brokers, dtype=np.int32)
        self.spike_multiplier = np.ones(self.num_brokers, dtype=np.float32)

        self.current_step = 0
        self.incoming_partition_load, self.incoming_broker_hint = self._generate_partition_load()

        observation = self._get_obs()
        info = {}
        return observation, info
    
    def _generate_partition_load(self):
        drift = self.np_random.normal(0,0.1,size=self.num_brokers)
        self.broker_baseline = np.clip(self.broker_baseline+drift, 1.0,20.0)

        spike_prob = 0.02

        for i in range(self.num_brokers):
            if self.spike_remaining[i] == 0 and self.np_random.random() < spike_prob:
                self.spike_remaining[i] = self.np_random.integers(3,8)
                self.spike_multiplier[i] = self.np_random.uniform(5.0,10.0)
        
        active_spike_brokers = self.spike_remaining > 0
        self.spike_remaining[active_spike_brokers]-=1

        target_broker = self.np_random.integers(0, self.num_brokers)
        base = self.broker_baseline[target_broker]
        multiplier = self.spike_multiplier[target_broker] if self.spike_remaining[target_broker] > 0 else 1.0
        load = base * multiplier + self.np_random.normal(0, 0.5)
        load = max(load, 0.5)

        return load, target_broker

    def _get_obs(self):
        return np.concatenate([
            self.broker_loads,
            self.broker_partition_counts,
            self.broker_baseline,
            self.spike_remaining.astype(np.float32),
            [self.incoming_partition_load]
        ]).astype(np.float32)
    
    def step(self, action):
        self.broker_loads[action] += self.incoming_partition_load
        self.broker_partition_counts[action] += 1

        reward=self._compute_reward()

        self.current_step += 1
        self.incoming_partition_load, self.incoming_broker_hint = self._generate_partition_load()

        terminated = False
        truncated = self.current_step >= self.episode_length

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def _compute_reward(self):
        load_std = np.std(self.broker_loads)
        return -load_std