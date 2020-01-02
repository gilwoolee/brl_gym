import numpy as np
import abc

"""
Abstract class
"""
class Expert(metaclass=abc.ABCMeta):
    def __init__(self, obs_dim, belief_dim):
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim

    @abc.abstractmethod
    def action(self, inputs, infos=None):
        raise NotImplementedError

    def _split_inputs(self, inputs):
        obs, bel = inputs[:, :self.obs_dim], inputs[:, self.obs_dim:]
        return obs, bel

    def __call__(self, inputs, infos=None):
        return self.action(inputs, infos)
