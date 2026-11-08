"""
Train two PPO agents on CartPole:

  1. BASELINE  - fixed default physics (no randomisation)
  2. DR        - domain-randomised physics (new values every reset)

Both are trained for the same number of timesteps, then saved to disk.
Episode rewards during training are logged with a Monitor so we can plot
learning curves later.

Run:
    python train.py
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from dr_env import DomainRandomizationCartPole

TIMESTEPS = 80_000
SEED = 0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def train_agent(name, randomize):
    print(f"\n=== Training {name} (randomize={randomize}) ===")
    env = DomainRandomizationCartPole(randomize=randomize, seed=SEED)
    env = Monitor(env, filename=os.path.join(LOG_DIR, name))

    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=0,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.0,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    model.save(os.path.join(OUT_DIR, f"ppo_{name}"))
    env.close()
    print(f"Saved model: ppo_{name}.zip")


if __name__ == "__main__":
    np.random.seed(SEED)
    train_agent("baseline", randomize=False)
    train_agent("dr", randomize=True)
    print("\nTraining complete. Now run:  python evaluate.py")
