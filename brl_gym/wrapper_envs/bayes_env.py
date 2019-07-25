import numpy as np

import gym
from gym.spaces import Box, Discrete

from brl_gym.wrapper_envs import WrapperEnv
from brl_gym.wrapper_envs.util import to_one_hot

class BayesEnv(WrapperEnv):
    """
    An environment with latent MDP estimator and a modifiable env.
    Belief over possible MDPs is augmented to observation.
    """
    def __init__(self, env, estimator, augment_to_obs=True):

        super(BayesEnv, self).__init__(env, estimator, augment_to_obs)

        # Augment with belief space
        if isinstance(env.observation_space, Box):
            lows = np.concatenate([
                env.observation_space.low,
                estimator.belief_low], axis=0)
            highs = np.concatenate([
                env.observation_space.high,
                estimator.belief_high], axis=0)
        else:
            lows = np.concatenate([
                np.zeros(env.observation_space.n),
                estimator.belief_low], axis=0)
            highs = np.concatenate([
                np.ones(env.observation_space.n),
                estimator.belief_high], axis=0)

        self.observation_space = Box(lows, highs, dtype=np.float32)
        self.belief_space = estimator.belief_space

    def _augment_observation(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        self.estimator.estimate(
                action, obs, **kwargs)
        belief = self.estimator.get_belief()

        if not isinstance(belief, np.ndarray):
            belief = np.array([belief])

        if isinstance(self.env.observation_space, Discrete):
            obs = to_one_hot(obs, self.env.observation_space.n)

        if not isinstance(obs, np.ndarray):
            obs = np.array([obs])

        kwargs['belief'] = belief
        if self.augment_to_obs:
            return np.concatenate([obs, belief], axis=0), kwargs
        else:

            return obs, kwargs

    def set_state(self, state):
        self.env.set_state(state)

    def get_state(self):
        return self.env.get_state()
