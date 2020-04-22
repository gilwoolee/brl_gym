import numpy as np
import torch
from brl_gym.wrapper_envs import BayesEnv
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv
from brl_gym.estimators.classic_control.bayes_continuous_cartpole_estimator import BayesContinuousCartpoleEstimator
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from brl_gym.estimators.estimator import Estimator
from gym.spaces import Box, Dict

import collections

# from brl_gym.scripts.continuous_cartpole.model import BayesFilterNet2

class BayesContinuousCartPoleEnv(BayesEnv):
    # Wrapper envs for mujoco envs
    def __init__(self, learned_bf=False):
        self.env = ContinuousCartPoleEnv(random_param=True)
        self.estimator = BayesContinuousCartpoleEstimator()

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
        belief = self.estimator.estimate(action, obs, **info)
        info['belief'] = belief

        obs = np.concatenate([obs, belief], axis=0)
        return obs, reward, done, info

class BayesContinuousCartPoleEnvLBF(ContinuousCartPoleEnv):
    # Wrapper envs for mujoco envs
    def __init__(self, **kwargs):
        self.env = ContinuousCartPoleEnv(random_param=True)
        self.estimator = LearnableBF(**kwargs)
        self.bel_input = collections.deque(maxlen=int(self.estimator.input_dim // 10) + 1)
        self.last_input = np.zeros(self.estimator.input_dim)

        self.hidden = None
        super(BayesContinuousCartPoleEnvLBF, self).__init__(self.env, self.estimator)

    def reset(self):
        obs = self.env.reset()
        bel = self.estimator.reset()
        self.bel_input = collections.deque(maxlen=int(self.estimator.input_dim // 10) + 1)
        self.bel_input.append(np.zeros(5)); self.bel_input.append(np.zeros(5))
        self.last_input = np.zeros(self.estimator.input_dim)
        bel = np.zeros(10)
        bel.fill(0.2)
        # TODO: Maybe get the belief here from inference
        # obs = np.concatenate([obs, bel], axis=0)
        # print ("b: ", obs.shape)
        self.hidden = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        feat = np.append(obs, action)
        self.bel_input.append(feat)

        # Estimate
        # TODO : Create estimate function in learned bf
        belief = self.estimator.estimate(self.bel_input, self.last_input)
        info['belief'] = belief
        info['params'] = np.array([self.env.masspole, self.env.length])
        # print ("ob: ", obs.shape, " bel: ", belief.shape)
        # obs = torch.from_numpy(obs).float().to('cuda')
        # obs = torch.cat([obs, belief])
        # obs = np.concatenate([obs, belief], axis=0)
        return obs, reward, done, info


class BayesContinuousCartPoleEnvLBF2(ContinuousCartPoleEnv):
    # Wrapper envs for mujoco envs
    def __init__(self, **kwargs):
        self.env = ContinuousCartPoleEnv(random_param=True)
        self.estimator = LearnableBF(**kwargs)
        self.bel_input = collections.deque(maxlen=int(self.estimator.input_dim // 10))

        self.prev_feat = None
        self.hidden = None
        super(BayesContinuousCartPoleEnvLBF2, self).__init__(self.env, self.estimator)

    def reset(self):
        obs = self.env.reset()
        bel = self.estimator.reset()
        self.prev_feat = np.concatenate((obs, [0.]))
        # self.bel_input = collections.deque(maxlen=int(self.estimator.input_dim // 10))
        self.bel_input.clear()
        # self.bel_input.append(np.concatenate((self.prev_feat, np.zeros(5))))
        # self.last_input = np.zeros(self.estimator.input_dim)

        # TODO: Maybe get the belief here from inference
        self.hidden = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        feat = np.append(obs, action)
        input_feat = np.concatenate((feat, (feat - self.prev_feat)))
        self.prev_feat = feat
        self.bel_input.append(input_feat)

        # Estimate
        belief = self.estimator.estimate2(self.bel_input)
        info['belief'] = belief
        info['params'] = np.array([self.env.masspole, self.env.length])
        return obs, reward, done, info



class LearnableBF(Estimator):
    def __init__(self, model, input_dim, belief_dim, nonlinear='relu'):
        self.input_dim = input_dim
        # output_dim = belief_dim
        self.hidden_state = None
        self.belief_dim = belief_dim
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = model
        self.model = self.model.to(self.device)
        self.temperature = 1
        self.belief_low = np.zeros(belief_dim)
        self.belief_high = np.ones(belief_dim)
        self.belief_space = belief_dim

    def reset(self):
        self.hidden_state = None
        self.belief = np.ones(self.belief_dim) / self.belief_dim
        self.belief = torch.from_numpy(self.belief)
        return self.belief

    def estimate(self, bel_input, last_input):
        # TODOL All of this should be tensors
        bel_input = np.array(bel_input)
        bel_input = np.concatenate((bel_input[1:,:], bel_input[1:,:] - bel_input[:-1, :]), axis=1)
        bel_input = np.concatenate((bel_input), axis=0)
        idx = np.arange(len(bel_input))
        np.put(last_input, idx, bel_input)
        final_input = torch.from_numpy(np.reshape(last_input, (1, 1, -1))).float().to(self.device)
        # print ("Fi : ", final_input.size())
        final_input = (final_input - self.model.means) / self.model.stds
        # print ("Fi2 : ", final_input.size())
        _, self.hidden_state, belief = self.model.get_belief(final_input, self.hidden_state)
        return belief[0][0]

    def estimate2(self, bel_input):
        bel_input = np.array(bel_input)
        feat_repeat = bel_input[0]
        bel_input = np.concatenate((bel_input), axis=0)
        rep_n = int(self.input_dim // 10 - len(bel_input) // 10)
        rep_feat = np.repeat(feat_repeat, rep_n)
        bel_input = np.concatenate((rep_feat, bel_input))
        final_input = torch.from_numpy(np.reshape(bel_input, (1, 1, -1))).float().to(self.device)
        final_input = (final_input - self.model.means) / self.model.stds
        _, self.hidden_state, belief = self.model.get_belief(final_input, self.hidden_state)
        return belief[0][0]
        # raise NotImplementedError

    def get_belief(self):
        return self.belief.copy()

    def forward(self, action, observation, **kwargs):
        if action is None:
            return self.reset()
        inp = np.concatenate([action, observation, [int(kwargs['done'])]], axis=0).ravel()
        inp = torch.Tensor(inp).float().to(self.device)

        inp = inp.reshape(1, 1, -1)
        output, self.hidden_state = self.model(inp, self.hidden_state)

        x = output / max(self.temperature, 1e-20)
        x = F.softmax(x, dim=2)

        self.belief = x.detach().cpu().numpy().ravel()
        return x, output, self.hidden_state

    def set_bayes_filter(self, model_path):
        self.model.load_last_model(model_path)

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

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
