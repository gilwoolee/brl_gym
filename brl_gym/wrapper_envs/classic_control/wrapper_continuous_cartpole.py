import numpy as np
import torch
from brl_gym.wrapper_envs import BayesEnv
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv
from brl_gym.estimators.classic_control.bayes_continuous_cartpole_estimator import BayesContinuousCartpoleEstimator
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from gym.spaces import Box, Dict

# from brl_gym.scripts.continuous_cartpole.model import BayesFilterNet2

class BayesContinuousCartPoleEnv(BayesEnv):
    # Wrapper envs for mujoco envs
    def __init__(self, learned_bf=False):
        self.env = ContinuousCartPoleEnv(random_param=True)
        if learned_bf:
            self.estimator = BayesFilterNet2(5, 2, 16)
            self.estimator.eval()
        else:
            self.estimator = BayesContinuousCartpoleEstimator()
        # self.model = BayesFilterNet2(5, 2, 16)
        # self.model.eval()
        # self.model.load_last_model("/home/rishabh/work/brl_gym/brl_gym/scripts/continuous_cartpole/data/2020-03-17_01-27-02/estimator_xx_checkpoints_mse")

        self.hidden = None
        super(BayesContinuousCartPoleEnv, self).__init__(self.env, self.estimator)

    def reset(self):
        obs = self.env.reset()
        bel = self.estimator.estimate(None, obs)
        obs = np.concatenate([obs, bel], axis=0)
        self.hidden = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Estimate
        # TODO : Create estimate function in learned bf
        # input_data = np.expand_dims(np.hstack((obs, action)), axis=0)
        # input_data = np.expand_dims(input_data, axis=0)
        # input_data = torch.Tensor(input_data)
        # with torch.no_grad():
        #     out, self.hidden, bel = self.model.get_belief(input_data, self.hidden)
        # bel = bel.cpu().data.numpy()
        # bel = bel.reshape(bel.shape[-1])
        belief = self.estimator.estimate(action, obs, **info)
        # print ("Belief: ", bel.shape)
        info['belief'] = belief

        obs = np.concatenate([obs, belief], axis=0)
        return obs, reward, done, info

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
