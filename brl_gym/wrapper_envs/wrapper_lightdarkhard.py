from brl_gym.wrapper_envs.bayes_env import BayesEnv
from brl_gym.envs.lightdarkhard import LightDarkHard
from brl_gym.estimators.ekf_lightdarkhard_estimator import EKFLightDarkHardEstimator

import gym
from gym import utils

class BayesLightDarkHard(BayesEnv):
    def __init__(self, reward_entropy=False):
        env = LightDarkHard()
        estimator = EKFLightDarkHardEstimator()
        self.reward_entropy = reward_entropy
        super(BayesLightDarkHard, self).__init__(env, estimator)

    def reset(self):
        obs = super().reset()
        self.cov = self.estimator.get_belief()[-1]
        return obs

    def step(self, action):
        obs, rew, done, info = super().step(action)
        cov = info["belief"][-1]
        #ent_rew = (1.0 - cov / (self.cov + 1e-3)) * 10.0
        ent_rew = (self.cov - cov) * 10.0
        self.cov = cov

        if self.reward_entropy:
            rew = rew + ent_rew #+ (10.0 - info["noise"])
        info['ent-rew'] = ent_rew
        return obs, rew, done, info




if __name__ == "__main__":
    print("=============== Bayes ===============")
    import numpy as np

    env = BayesLightDarkHard(reward_entropy=True)
    obs = env.reset()
    print (obs)
    for _ in range(10):
        print(   )
        action = env.action_space.sample()
        obs, rew, _, inf = env.step(np.array([1.0,0.0]))
        print('obs    ', np.around(obs, 2))

        print('bel    ', np.around(inf['belief'],2))
        print('ent-rew', inf['ent-rew'])
        print('goaldir', env.env.goal - env.env.x)
        print('rew    ', rew)
