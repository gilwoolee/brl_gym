import numpy as np

from brl_gym.wrapper_envs import BayesEnv
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv
from brl_gym.estimators.classic_control.bayes_continuous_cartpole_estimator import BayesContinuousCartpoleEstimator
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from gym.spaces import Box, Dict

class BayesContinuousCartPoleEnv(BayesEnv):
    # Wrapper envs for mujoco envs
    def __init__(self):
        self.env = ContinuousCartPoleEnv(random_param=True)
        self.estimator = BayesContinuousCartpoleEstimator()
        super(BayesContinuousCartPoleEnv, self).__init__(self.env, self.estimator)

    def reset(self):
        obs = self.env.reset()
        bel = self.estimator.estimate(None, obs)
        obs = np.concatenate([obs, bel], axis=0)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Estimate
        belief = self.estimator.estimate(action, obs, **info)
        info['belief'] = belief

        obs = np.concatenate([obs, belief], axis=0)
        return obs, reward, done, info



class ExplicitBayesContinuousCartPoleEnv(ExplicitBayesEnv):
    def __init__(self):
        self.env = ContinuousCartPoleEnv(random_param=True)
        self.estimator = BayesContinuousCartpoleEstimator()
        super(ExplicitBayesContinuousCartPoleEnv, self).__init__(self.env, self.estimator)

        self.observation_space = Dict(
            {"obs": self.env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = self.env.observation_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Estimate
        bel = self.estimator.estimate(action, obs, **info)
        info['belief'] = bel

        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        obs = self.env.reset()
        bel = self.estimator.estimate(None, obs)
        return {'obs':obs, 'zbel':bel}

if __name__ == "__main__":
    env = ExplicitBayesContinuousCartPoleEnv()
    print(env.reset())
    for _ in range(200):
        obs, _, _, _ = env.step(env.action_space.sample())
        print(np.around(obs['obs'],1), np.around(obs['zbel'], 1))

    import IPython; IPython.embed()
