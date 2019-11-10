import numpy as np

import gym
from gym import utils
from gym.spaces import Box, Discrete, Dict


class ExplicitBayesEnv(gym.Env, utils.EzPickle):
    """
    An environment with latent MDP estimator and a modifiable env.
    The main difference from BayesEnv is that (obs, bel) is returned as a tuple
    """
    def __init__(self, env, estimator):
        self.env = env
        self.estimator = estimator
        self.action_space = env.action_space
        self.belief_space = estimator.belief_space
        # self.observation_space = Dict({"obs": env.observation_space, "bel": estimator.belief_space})
        # self.internal_observation_space = env.observation_space


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        bel, info = self._update_belief(
                                        action,
                                        obs,
                                        **info)
        self.last_obs = (obs, bel)
        return {'obs':obs, 'zbel':bel}, reward, done, info

    """
    def reset(self):
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        self.last_obs = (obs, bel)
        return {'obs':obs, 'zbel':bel}
    """

    def _update_belief(self,
                             action,
                             obs,
                             **kwargs):
        raise NotImplementedError

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def set_state(self, state):
        self.env.set_state(state)

    def get_state(self):
        return self.env.get_state()
