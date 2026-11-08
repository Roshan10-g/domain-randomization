"""
Domain Randomisation environment wrapper for CartPole-v1.

Idea
----
On every reset, we randomise the physics parameters of the pole-cart system
(cart mass, pole mass, pole length, gravity). An agent trained on this
*distribution* of worlds is forced to learn a robust policy instead of
over-fitting to one specific physics setting. At test time we evaluate on
parameter values the agent has never seen.
"""

import numpy as np
import gymnasium as gym


# Default CartPole physics (from Gymnasium source) for reference / baseline.
DEFAULTS = {
    "gravity": 9.8,
    "masscart": 1.0,
    "masspole": 0.1,
    "length": 0.5,  # actually half the pole's length
}


class DomainRandomizationCartPole(gym.Wrapper):
    """Randomise CartPole physics on every reset.

    Parameters
    ----------
    ranges : dict | None
        Mapping of physics attribute -> (low, high). If None, uses a default
        training range. Each reset samples a fresh value uniformly.
    randomize : bool
        If False, the environment keeps the default physics (used for the
        baseline agent).
    seed : int | None
        Seed for the internal parameter sampler (independent of env seeding).
    """

    TRAIN_RANGES = {
        "gravity": (5.0, 15.0),
        "masscart": (0.5, 2.0),
        "masspole": (0.05, 0.5),
        "length": (0.3, 0.7),
    }

    def __init__(self, ranges=None, randomize=True, seed=None):
        env = gym.make("CartPole-v1")
        super().__init__(env)
        self.ranges = ranges if ranges is not None else self.TRAIN_RANGES
        self.randomize = randomize
        self._rng = np.random.default_rng(seed)
        self.current_params = dict(DEFAULTS)

    def _sample_params(self):
        params = {}
        for key, (low, high) in self.ranges.items():
            params[key] = float(self._rng.uniform(low, high))
        return params

    def _apply_params(self, params):
        # The underlying CartPole stores physics as attributes on the
        # unwrapped env. We write to them directly.
        core = self.env.unwrapped
        core.gravity = params.get("gravity", core.gravity)
        core.masscart = params.get("masscart", core.masscart)
        core.masspole = params.get("masspole", core.masspole)
        core.length = params.get("length", core.length)
        # Recompute dependent quantities the way CartPole does internally.
        core.total_mass = core.masspole + core.masscart
        core.polemass_length = core.masspole * core.length
        self.current_params = params

    def reset(self, **kwargs):
        if self.randomize:
            self._apply_params(self._sample_params())
        return self.env.reset(**kwargs)

    def set_fixed_params(self, params):
        """Force a specific physics setting (used for evaluation)."""
        self.randomize = False
        self._apply_params(params)


def make_train_env(randomize=True, seed=None):
    """Factory used by stable-baselines3."""
    def _init():
        return DomainRandomizationCartPole(randomize=randomize, seed=seed)
    return _init
