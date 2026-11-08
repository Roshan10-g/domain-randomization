"""
Visual demo — watch the trained agent balance the pole!

Opens a window showing the CartPole simulation with the DR agent in action.
Press Ctrl+C in terminal to stop.

Run:
    python demo.py
"""

import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

# Load the trained DR agent
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ppo_dr.zip")

print("Loading trained agent...")
model = PPO.load(MODEL_PATH)

# Create environment with render mode
env = gym.make("CartPole-v1", render_mode="human")

print("\n" + "="*50)
print("  DOMAIN RANDOMIZATION DEMO")
print("  Watch the trained agent balance the pole!")
print("  Close the window or Ctrl+C to stop.")
print("="*50 + "\n")

# Run episodes
episode = 0
while True:
    episode += 1
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.02)  # Slow down slightly for visibility
    
    print(f"Episode {episode}: Reward = {total_reward:.0f}")

env.close()
