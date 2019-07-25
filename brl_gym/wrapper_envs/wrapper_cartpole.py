import numpy as np

from brl_gym.wrapper_envs import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteParamEnvSampler, UniformSampler
from brl_gym.estimators.param_env_discrete_estimator import ParamEnvDiscreteEstimator
from brl_gym.envs.classic_control.cartpole import CartPoleEnv
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from gym.spaces import Box, Dict

class BayesCartPoleEnv(BayesEnv):
    # Wrapper envs for mujoco envs
    def __init__(self):
        env = CartPoleEnv()
        self.estimator = ParamEnvDiscreteEstimator(env, discretization=2)

        self.env_sampler = DiscreteParamEnvSampler(env, 2)
        super(BayesCartPoleEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

    def reset(self):
        self.env = self.env_sampler.sample()
        return super().reset()

    def step(self, action):
        prev_belief = self.estimator.get_belief()
        prev_state = self.env.get_state()
        obs, reward, done, info = self.env.step(action)
        info['prev_state'] = prev_state
        info['curr_state'] = self.env.get_state()

        # Estimate
        self.estimator.estimate(action, obs, **info)
        belief = self.estimator.get_belief()
        info['belief'] = belief

        obs = np.concatenate([obs, belief], axis=0)
        return obs, reward, done, info



class ExplicitBayesCartPoleEnv(ExplicitBayesEnv):
    def __init__(self):
        env = CartPoleEnv()
        self.estimator = ParamEnvDiscreteEstimator(env, discretization=2)

        self.env_sampler = DiscreteParamEnvSampler(env, 2)
        super(ExplicitBayesCartPoleEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = env.observation_space
        self.env = env

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
        prev_state = self.env.get_state()
        obs, reward, done, info = self.env.step(action)
        info['prev_state'] = prev_state
        info['curr_state'] = self.env.get_state()

        bel, info = self._update_belief(
                                        action,
                                        obs,
                                        **info)
        true_param = self.env.get_params()
        # mass = true_param['masspole']
        length = true_param['length']

        exp1 = np.argwhere(self.env_sampler.param_sampler_space['length'] == length)[0,0]
        # exp2 = np.argwhere(self.env_sampler.param_sampler_space['masspole'] == mass)[0,0]
        exp_id = exp1
        info['expert'] = exp_id

        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        self.env = self.env_sampler.sample()
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        self.last_obs = (obs, bel)
        return {'obs':obs, 'zbel':bel}

if __name__ == "__main__":
    env = ExplicitBayesCartPoleEnv()
    print(env.reset())
    for _ in range(200):
        print(env.step(env.action_space.sample())[0])

    import IPython; IPython.embed()
