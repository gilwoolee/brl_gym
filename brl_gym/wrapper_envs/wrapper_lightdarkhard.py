from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.envs.lightdarkhard import LightDarkHard
from brl_gym.estimators.ekf_lightdarkhard_estimator import EKFLightDarkHardEstimator

import gym
from gym import utils

class BayesLightDarkHard(BayesEnv):
    def __init__(self):
        env = LightDarkHard()
        estimator = EKFLightDarkHardEstimator()
        super(BayesLightDarkHard, self).__init__(env, estimator)


if __name__ == "__main__":
    print("=============== Bayes ===============")

    env = BayesLightDarkHard()
    obs = env.reset()
    print (obs)
    for _ in range(10):
        action = env.action_space.sample()
        print(env.step(action))

