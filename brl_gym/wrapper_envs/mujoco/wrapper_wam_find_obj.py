from brl_gym.wrapper_envs.bayes_env import BayesEnv

from brl_gym.envs.mujoco.wam_find_obj import WamFindObjEnv
from brl_gym.estimators.mujoco.ekf_wam_find_obj_estimator import EKFWamFindObjEstimator
import gym
from gym import utils
from gym.spaces import Box
import numpy as np


class BayesWamFindObj(BayesEnv):
    def __init__(self):
        env = WamFindObjEnv()
        estimator = EKFWamFindObjEstimator(env.action_space)
        super(BayesWamFindObj, self).__init__(env, estimator)

        #obs_space = env.observation_space

        #self.observation_space = Box(self.estimator.belief_low,
        #                             self.estimator.belief_high, dtype=np.float32)
        self.viewer = None

    def _augment_observation(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        belief = self.estimator.estimate(
                action, obs, **kwargs)
        obs = np.concatenate([obs, belief], axis=0)
        return obs, kwargs


class MLEWamFindObj(BayesEnv):
    def __init__(self):
        env = WamFindObjEnv()
        estimator = EKFWamFindObjEstimator(env.action_space)
        super(MLEWamFindObj, self).__init__(env, estimator)

        obs_space = env.observation_space

        self.observation_space = Box(
                np.concatenate([obs_space.low, self.estimator.belief_low[:2]]),
                np.concatenate([obs_space.high, self.estimator.belief_high[:2]]),
                    dtype=np.float32)

        self.viewer = None

    def _augment_observation(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        belief = self.estimator.estimate(
                action, obs, **kwargs)
        obs = np.concatenate([obs, belief[:2]], axis=0)
        return obs, kwargs


if __name__ == "__main__":
    print("=============== Bayes ===============")
    env = MLEWamFindObj()
    obs = env.reset()
    print(env.observation_space)
    for _ in range(10):
        o, r, d, _ = env.step(env.action_space.sample())
        print(o, r)
    import IPython; IPython.embed()

