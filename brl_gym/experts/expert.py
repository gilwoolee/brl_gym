import numpy as np

"""
Abstract class
"""
class Expert:
    def __init__(self, obs_dim, belief_dim):
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim

    @abstractmethod
    def action(self, inputs, infos):
        raise NotImplementedError

    def _split_inputs(self, inputs):
        obs, bel = inputs[:, :obs_dim], inputs[:, obs_dim:]
        return obs, bel
