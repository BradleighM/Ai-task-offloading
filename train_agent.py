import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class EdgeEnv(gym.Env):
    """
    A virtual world for our AI to learn in.
    State: [CPU_Usage (%), Network_Latency (ms), Task_Complexity]
    Action: 0 (Local), 1 (Remote/Offload)
    """
    def __init__(self):
        super(EdgeEnv, self).__init__()
        # Action: 0 or 1
        self.action_space = spaces.Discrete(2)
        # Observation: CPU (0-100), Latency (0-500ms), Task (1-10)
        self.observation_space = spaces.Box(low=0, high=500, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with random conditions
        self.state = np.array([np.random.uniform(0, 100), 
                               np.random.uniform(10, 300), 
                               np.random.uniform(1, 10)], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        cpu, latency, complexity = self.state
        
        # Calculate "Cost"
        if action == 0: # Local
            cost = (complexity * 20) + (cpu * 0.5) # High CPU makes local processing "expensive"
            time_taken = complexity * 0.1
        else: # Remote
            cost = latency * 0.5 # High latency makes offloading "expensive"
            time_taken = (latency / 1000) + 0.05
        
        # Reward is the negative cost (we want to minimize cost)
        reward = -cost
        
        # Generate next random state
        self.state = np.array([np.random.uniform(0, 100), 
                               np.random.uniform(10, 300), 
                               np.random.uniform(1, 10)], dtype=np.float32)
        
        done = False # Keep playing
        truncated = False
        return self.state, reward, done, truncated, {}

# --- Training Code ---
if __name__ == "__main__":
    print("🚀 Creating Environment...")
    env = EdgeEnv()

    print("🧠 Training AI Brain (PPO Algorithm)...")
    # PPO is a state-of-the-art RL algorithm
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train it for 20,000 "steps" or "decisions"
    model.learn(total_timesteps=20000)

    print("💾 Saving Model...")
    model.save("edge_ai_model")
    print("✅ Done! 'edge_ai_model.zip' created.")


