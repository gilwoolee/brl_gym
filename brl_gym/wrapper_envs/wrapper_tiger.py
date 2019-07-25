from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.wrapper_envs.mle_env import MLEEnv

from brl_gym.envs.tiger import Tiger
from brl_gym.estimators.bayes_tiger_estimator import BayesTigerEstimator
from brl_gym.wrapper_envs.util import to_one_hot
from brl_gym.envs.tiger import Action, TigerLocation

import gym
from gym import utils
from gym.spaces import Box, Dict

import numpy as np

class BayesTiger(BayesEnv):
    def __init__(self):
        env = Tiger()
        estimator = BayesTigerEstimator()
        super(BayesTiger, self).__init__(env, estimator)

        """
        self.observation_space = Box(
                estimator.belief_low,
                estimator.belief_high,
                dtype=np.float32)
        """

    """
    def _augment_observation(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        self.estimator.estimate(
                action, obs, **kwargs)
        belief = self.estimator.get_belief()
        return belief, kwargs
    """


class ExplicitBayesTiger(ExplicitBayesEnv):
    def __init__(self, tiger=None):
        env = Tiger(tiger=tiger)
        estimator = BayesTigerEstimator()
        super(ExplicitBayesTiger, self).__init__(env, estimator)

        obs_space = Box(np.zeros(3), np.ones(3), dtype=np.float32)
        self.observation_space = Dict({"obs": obs_space, "zbel": estimator.belief_space})
        # self.observation_space = Dict({"obs": env.observation_space, "bel": estimator.belief_space})
        # self.belief_space = estimator.belief_space
        self.internal_observation_space = obs_space

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
        obs, reward, done, info = self.env.step(action)
        bel, info = self._update_belief(
                                        action,
                                        obs,
                                        **info)

        # obs = to_one_hot(obs, 3)
        obs = np.array([0, 0, 1])
        self.last_obs = (obs, bel)
        info['expert'] = self.env.tiger
        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        obs = to_one_hot(obs, 3)
        self.last_obs = (obs, bel)
        return {'obs':obs, 'zbel':bel}


class MLETiger(MLEEnv):
    def __init__(self):
        env = Tiger()
        estimator = BayesTigerEstimator()
        super(MLETiger, self).__init__(env, estimator)


def get_value(tiger, action):
    if tiger == TigerLocation.LEFT:
        if action == Action.OPEN_RIGHT:
            return 1
        elif action == Action.OPEN_LEFT:
            return -10
        else:
            return -0.1
    else:
        if action == Action.OPEN_RIGHT:
            return -10
        elif action == Action.OPEN_LEFT:
            return 1
        else:
            return -0.1

def collect_batches(n_iterations):
    experiences = []
    from brl_gym.qmdps.tiger_qmdp import TigerQMDPQFunctionNP

    agent = TigerQMDPQFunctionNP()

    for i in range(2):
        observations = []
        values = []
        new_observations = []
        rewards = []
        dones = []
        actions = []

        for _ in range(n_iterations):
            env = ExplicitBayesTiger(i)
            o = env.reset()

            done = False
            while not done:
                # action = Action.OPEN_RIGHT if i == TigerLocation.LEFT else Action.OPEN_LEFT
                # if np.random.rand() < 0.05:
                #action = env.action_space.sample()
                action = agent.step(o['obs'].reshape(1,-1), o['zbel'].reshape(1,-1))

                values += [get_value(i, action)]
                observations += [np.concatenate([o['obs'], o['zbel']])]
                o, r, done, _ = env.step(action)
                actions += [action]
                new_observations += [np.concatenate([o['obs'], o['zbel']])]
                rewards += [r]
                dones += [done]

        experiences += [(np.array(observations), np.array(values), np.array(actions),
                 np.array(rewards).reshape(1, -1), np.array(new_observations), dones)]

    return experiences

if __name__ == "__main__":
    print("=============== ExplicitBayes ===============")
    env = ExplicitBayesTiger()
    obs = env.reset()
    print (obs)
    for _ in range(10):
        print(env.step(0))

    print("=============== Bayes ===============")
    env = BayesTiger()
    obs = env.reset()
    print (obs)
    for _ in range(10):
        print(env.step(0))

    print("=============== MLE   ===============")
    env = MLETiger()
    obs = env.reset()
    print (obs)
    for _ in range(10):
        print(env.step(0))
