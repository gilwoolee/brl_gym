from brl_gym.wrapper_envs.wrapper_env import WrapperEnv

from brl_gym.envs.mujoco.wam_find_obj import WamFindObjEnv
from brl_gym.estimators.mujoco.ekf_wam_find_obj_estimator import EKFWamFindObjEstimator
import gym
from gym import utils
from gym.spaces import Box
import numpy as np


class BayesWamFindObj(WrapperEnv):
    def __init__(self):
        env = WamFindObjEnv()
        estimator = EKFWamFindObjEstimator(env.action_space)
        super(BayesWamFindObj, self).__init__(env, estimator)
        obs_space = env.observation_space

        self.observation_space = Box(self.estimator.belief_low,
                                     self.estimator.belief_high, dtype=np.float32)
        self.viewer = None

    def _augment_observation(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        belief = self.estimator.estimate(
                action, obs, **kwargs)
        return belief, kwargs



if __name__ == "__main__":
    print("=============== Bayes ===============")
    env = BayesWamFindObj()
    obs = env.reset()
    for _ in range(500):
        o, r, d, _ = env.step(env.action_space.sample())
        print(o, r)
    import IPython; IPython.embed()

