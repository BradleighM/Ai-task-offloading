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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.state = self._generate_state()
        return self.state, {}

    def _generate_state(self):
        self.time_step += 1
        
        # Base latency with a periodic spike (sine wave)
        base_latency = 50 + 40 * np.sin(self.time_step / 10.0) 
        # Add random spikes to simulate sudden network drops
        if np.random.rand() < 0.1:
            base_latency += 200
            
        latency = np.clip(base_latency, 10, 500)
        
        # CPU with slow drift
        cpu = 50 + 30 * np.sin(self.time_step / 25.0)
        cpu = np.clip(cpu, 0, 100)
        
        complexity = np.random.uniform(1, 10)
        
        # Server CPU with slow drift
        server_cpu = 50 + 40 * np.cos(self.time_step / 30.0)
        server_cpu = np.clip(server_cpu, 0, 100)
        
        return np.array([cpu, latency, complexity, server_cpu], dtype=np.float32)

    def step(self, action):
        cpu, latency, complexity, server_cpu = self.state
        
        # Calculate "Cost"
        if action == 0: # Local
            cost = (complexity * 20) + (cpu * 0.5) # High CPU makes local processing "expensive"
            time_taken = complexity * 0.1
        else: # Remote
            cost = (latency * 0.5) + (server_cpu * 1.0) # High latency or busy server makes offloading "expensive"
            time_taken = (latency / 1000) + 0.05 + (server_cpu / 100.0)
        
        # Reward is the negative cost (we want to minimize cost)
        reward = -cost
        
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
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    
    # Train it for 50,000 "steps" or "decisions"
    model.learn(total_timesteps=50000)

    print("💾 Saving Model...")
    model.save("edge_ai_model")
    print("✅ Done! 'edge_ai_model.zip' created.")
