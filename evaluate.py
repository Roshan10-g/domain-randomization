"""
Evaluate the BASELINE and DR agents on UNSEEN physics parameters.

We build a grid of out-of-distribution test settings (heavier poles,
different gravity, longer poles) that neither agent saw as a fixed world,
then measure average episode return over several episodes for each setting.

Produces:
  - Console table of mean returns
  - results/generalization_bar.png   (DR vs baseline across test worlds)
  - results/learning_curves.png      (training reward over time)

Run (after train.py):
    python evaluate.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from dr_env import DomainRandomizationCartPole, DEFAULTS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(OUT_DIR, "logs")
RES_DIR = os.path.join(OUT_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

N_EVAL_EPISODES = 20

# Unseen test worlds: deliberately at / beyond the edges of training range.
TEST_WORLDS = {
    "low_gravity":   {**DEFAULTS, "gravity": 4.0},
    "high_gravity":  {**DEFAULTS, "gravity": 18.0},
    "heavy_pole":    {**DEFAULTS, "masspole": 0.8},
    "heavy_cart":    {**DEFAULTS, "masscart": 3.0},
    "long_pole":     {**DEFAULTS, "length": 0.9},
    "default":       {**DEFAULTS},
}


def evaluate(model, params, n_episodes=N_EVAL_EPISODES):
    env = DomainRandomizationCartPole(randomize=False)
    env.set_fixed_params(params)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def plot_learning_curves():
    """Read Monitor CSV logs and plot smoothed reward vs timestep."""
    plt.figure(figsize=(8, 5))
    for name in ["baseline", "dr"]:
        path = os.path.join(LOG_DIR, f"{name}.monitor.csv")
        if not os.path.exists(path):
            continue
        # Monitor CSV: first line is a comment, header on line 2.
        data = np.genfromtxt(path, delimiter=",", skip_header=2)
        if data.ndim == 1 or len(data) == 0:
            continue
        rewards = data[:, 0]
        timesteps = np.cumsum(data[:, 1])
        # Simple moving average for readability.
        w = max(1, len(rewards) // 30)
        smooth = np.convolve(rewards, np.ones(w) / w, mode="valid")
        plt.plot(timesteps[w - 1:], smooth, label=name)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (smoothed)")
    plt.title("Learning curves: baseline vs domain randomisation")
    plt.legend()
    plt.grid(alpha=0.3)
    out = os.path.join(RES_DIR, "learning_curves.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    base_path = os.path.join(OUT_DIR, "ppo_baseline.zip")
    dr_path = os.path.join(OUT_DIR, "ppo_dr.zip")
    if not (os.path.exists(base_path) and os.path.exists(dr_path)):
        raise SystemExit("Models not found. Run `python train.py` first.")

    baseline = PPO.load(base_path)
    dr = PPO.load(dr_path)

    worlds = list(TEST_WORLDS.keys())
    base_means, dr_means = [], []

    print(f"\n{'World':<14}{'Baseline':>18}{'Domain Rand.':>18}")
    print("-" * 50)
    for w in worlds:
        bm, bs = evaluate(baseline, TEST_WORLDS[w])
        dm, ds = evaluate(dr, TEST_WORLDS[w])
        base_means.append(bm)
        dr_means.append(dm)
        print(f"{w:<14}{bm:>10.1f} +/-{bs:<5.1f}{dm:>10.1f} +/-{ds:<5.1f}")

    print("-" * 50)
    print(f"{'AVERAGE':<14}{np.mean(base_means):>14.1f}"
          f"{np.mean(dr_means):>18.1f}")

    # Bar chart comparison.
    x = np.arange(len(worlds))
    width = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, base_means, width, label="Baseline (fixed physics)")
    plt.bar(x + width / 2, dr_means, width, label="Domain randomisation")
    plt.xticks(x, worlds, rotation=20)
    plt.ylabel("Mean return over 20 episodes")
    plt.title("Generalisation to UNSEEN physics (higher = better)")
    plt.axhline(500, color="gray", ls="--", lw=1, label="max possible (500)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    out = os.path.join(RES_DIR, "generalization_bar.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")

    plot_learning_curves()


if __name__ == "__main__":
    main()
