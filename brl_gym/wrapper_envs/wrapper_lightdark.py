from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.wrapper_envs.mle_env import MLEEnv
from brl_gym.envs.lightdark import LightDark
from brl_gym.estimators.ekf_lightdark_estimator import EKFLightDarkEstimator

import gym
from gym import utils

class BayesLightDark(BayesEnv):
    def __init__(self):
        env = LightDark()
        estimator = EKFLightDarkEstimator()
        super(BayesLightDark, self).__init__(env, estimator)


class MLELightDark(MLEEnv):
    def __init__(self):
        env = LightDark()
        estimator = EKFLightDarkEstimator()
        super(MLELightDark, self).__init__(env, estimator)

if __name__ == "__main__":
    print("=============== Bayes ===============")

    env = BayesLightDark()
    obs = env.reset()
    print (obs)
    for _ in range(10):
        print(env.step(0))

    print("=============== MLE   ===============")

    env = MLELightDark()
    obs = env.reset()
    print (obs)
    for _ in range(10):
        print(env.step(0))
