import numpy as np

from brl_gym.wrapper_envs import BayesEnv
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv
from brl_gym.estimators.classic_control.bayes_continuous_cartpole_estimator import BayesContinuousCartpoleEstimator
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from gym.spaces import Box, Dict
from matplotlib import pyplot as plt

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
        self.t = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Estimate
        belief = self.estimator.estimate(action, obs, **info)
        info['belief'] = belief

        obs = np.concatenate([obs, belief], axis=0)
        self.t += 1
        return obs, reward, done, info

    def _visualize(self, **kwargs):
        img = self.env.render(mode='rgb_array')
        plt.clf()
        plt.imshow(img)
        plt.savefig('imgs/trial3/bayes_cartpole_{}.png'.format(self.t), bbox_inches='tight')


class MLEContinuousCartPoleEnv(BayesEnv):
    def __init__(self):
        self.env = ContinuousCartPoleEnv(random_param=True)
        self.estimator = BayesContinuousCartpoleEstimator()
        super(MLEContinuousCartPoleEnv, self).__init__(self.env, self.estimator)
        self.observation_space = Box(
                np.concatenate([self.env.observation_space.low, self.env.param_space_flat.low]),
                np.concatenate([self.env.observation_space.high, self.env.param_space_flat.high]),
                dtype=np.float32)
        self.internal_observation_space = self.env.observation_space

    def _get_obs(self, obs):
        params = self.estimator.get_best_params()
        obs = np.concatenate([obs, [params['length'], params['masscart']]])
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.estimator.estimate(action, obs, **info)
        return self._get_obs(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.estimator.estimate(None, obs)
        return self._get_obs(obs)


if __name__ == "__main__":
    env = MLEContinuousCartPoleEnv()
    print(env.reset())
    for _ in range(10):
        obs, _, _, _ = env.step(env.action_space.sample())
        print(obs)

    import IPython; IPython.embed()
