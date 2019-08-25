import numpy as np
from brl_gym.estimators.bayes_doors_estimator import BayesDoorsEstimator
from brl_gym.envs.mujoco.doors import DoorsEnv
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler

from gym.spaces import Box, Dict
from gym import utils

class ExplicitBayesDoorsEnv(ExplicitBayesEnv, utils.EzPickle):
    def __init__(self, reset_params=True):
        
        self.num_doors = 4
        self.num_cases = 2**self.num_doors
        self.cases =  ['{{:0{}b}}'.format(self.num_doors).format(x) \
                      for x in range(self.num_cases)]
        self.cases_np = [np.array([int(x) for x in case]) for case in self.cases]

        envs = []
        for case in self.cases_np:
            env = DoorsEnv()
            env.open_doors = case.astype(np.bool)
            envs += [env]

        self.estimator = BayesDoorsEstimator()
        
        self.env_sampler = DiscreteEnvSampler(envs)
        super(ExplicitBayesDoorsEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params
        utils.EzPickle.__init__(self)

    def _update_belief(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        self.estimator.estimate(
                action, obs, **kwargs)
        belief = self.estimator.get_belief()
        return belief, kwargs

    def step(self, action):
        prev_state = self.env.get_state().copy()
        obs, reward, done, info = self.env.step(action)
        info['prev_state'] = prev_state
        info['curr_state'] = self.env.get_state()

        bel, info = self._update_belief(
                                        action,
                                        obs,
                                        **info)

        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        if self.reset_params:
            self.env = self.env_sampler.sample()
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        
        return {'obs':obs, 'zbel':bel}


if __name__ == "__main__":
    env = ExplicitBayesDoorsEnv()
    for _ in range(10):
        env.reset()
        print(env.env.open_doors)

