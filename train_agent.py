import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import RecurrentPPO

class EdgeEnv(gym.Env):
    """
    A virtual world for our AI to learn in, with temporal dynamics.
    State: [CPU_Usage (%), Network_Latency (ms), Task_Complexity, Server_CPU (%)]
    Action: 0 (Local), 1 (Remote/Offload)
    """
    def __init__(self):
        super(EdgeEnv, self).__init__()
        # Action: 0 or 1
        self.action_space = spaces.Discrete(2)
        # Observation: Client CPU (0-100), Latency (0-500ms), Task (1-10), Server CPU (0-100)
        self.observation_space = spaces.Box(low=0, high=500, shape=(4,), dtype=np.float32)
        self.time_step = 0
        self.current_latency = 50.0
        self.current_cpu = 50.0
        self.current_server_cpu = 50.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.current_latency = np.random.uniform(10, 500)
        self.current_cpu = np.random.uniform(0, 100)
        self.current_server_cpu = np.random.uniform(0, 100)
        self.state = self._generate_state()
        return self.state, {}

    def _generate_state(self):
        self.time_step += 1
        
        # Random walk for temporal dynamics
        self.current_latency += np.random.normal(0, 30)
        self.current_latency = np.clip(self.current_latency, 10, 500)
        
        self.current_cpu += np.random.normal(0, 10)
        self.current_cpu = np.clip(self.current_cpu, 0, 100)
        
        self.current_server_cpu += np.random.normal(0, 10)
        self.current_server_cpu = np.clip(self.current_server_cpu, 0, 100)
        
        complexity = np.random.uniform(1, 10)
        
        # Occasional latency spikes
        lat = self.current_latency
        if np.random.rand() < 0.1:
            lat = 500.0
        
        return np.array([self.current_cpu, lat, complexity, self.current_server_cpu], dtype=np.float32)

    def step(self, action):
        cpu, latency, complexity, server_cpu = self.state
        
        # Calculate "Cost"
        cost_local = (complexity * 20) + (cpu * 0.5)
        cost_remote = (latency * 0.5) + (server_cpu * 1.0)
        
        if action == 0: # Local
            cost = cost_local
            time_taken = complexity * 0.1
        else: # Remote
            cost = cost_remote
            time_taken = (latency / 1000) + 0.05 + (server_cpu / 100.0)
        
        # Classification reward for RL
        if cost <= min(cost_local, cost_remote):
            reward = 1.0
        else:
            reward = -1.0
        
        # Generate next dynamic state
        self.state = self._generate_state()
        
        done = False # Keep playing
        truncated = False
        return self.state, reward, done, truncated, {}

# --- Training Code ---
if __name__ == "__main__":
    print("🚀 Creating Environment...")
    env = EdgeEnv()

    print("🧠 Training AI Brain (RecurrentPPO Algorithm)...")
    # RecurrentPPO uses an LSTM layer to track history
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, learning_rate=0.001)
    
    # Train it for 10,000 "steps" or "decisions" to ensure full coverage
    model.learn(total_timesteps=10000)

    print("💾 Saving Model...")
    model.save("edge_ai_model")
    print("✅ Done! 'edge_ai_model.zip' created.")
