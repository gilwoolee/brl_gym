import numpy as np

import gym
from gym import utils
from gym.spaces import Box, Discrete


class WrapperEnv(gym.Env, utils.EzPickle):
    """
    An environment with latent MDP estimator and a modifiable env.
    Useful feature (belief, param-estimate) is augmented to the observation
    """
    def __init__(self, env, estimator, augment_to_obs=True):

        self.env = env
        self.estimator = estimator
        self.augment_to_obs = augment_to_obs
        self.action_space = env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        augmented_obs, info = self._augment_observation(
                                        action,
                                        obs,
                                        **info)
        self.last_obs = augmented_obs
        return augmented_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.estimator.reset()
        obs, _ = self._augment_observation(action=None, obs=obs)
        self.last_obs = obs
        return obs

    def _augment_observation(self,
                             action,
                             obs,
                             **kwargs):
        raise NotImplementedError

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, mode='human'):
        self.env.render()
