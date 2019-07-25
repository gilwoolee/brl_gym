import numpy as np

from brl_gym.wrapper_envs import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteParamEnvSampler, UniformSampler
from brl_gym.estimators.param_env_discrete_estimator import ParamEnvDiscreteEstimator
from gym.envs.mujoco.humanoid_pushing import HumanoidPushingEnv
from brl_gym.wrapper_envs.explicit_bayes_env import ExplicitBayesEnv
from gym.spaces import Box, Dict
from baselines.common.math_util import discount

discretization=1

class BayesHumanoidPushingEnv(BayesEnv):
    # Wrapper envs for mujoco envs
    def __init__(self, reset_params=True):
        env = HumanoidPushingEnv()
        self.estimator = ParamEnvDiscreteEstimator(env, discretization=discretization)

        self.env_sampler = DiscreteParamEnvSampler(env, discretization)
        super(BayesHumanoidPushingEnv, self).__init__(env, self.estimator)
        self.nominal_env = env
        self.reset_params = reset_params

    def reset(self):
        if self.reset_params:
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



class ExplicitBayesHumanoidPushingEnv(ExplicitBayesEnv):
    def __init__(self, reset_params=True):
        env = HumanoidPushingEnv()
        self.estimator = ParamEnvDiscreteEstimator(env, discretization=discretization)

        self.env_sampler = DiscreteParamEnvSampler(env, discretization)
        self.env_sampler.param_space['friction']
        super(ExplicitBayesHumanoidPushingEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params

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
        friction = true_param['friction']

        exp1 = np.argwhere(self.env_sampler.param_sampler_space['friction'] == friction)[0,0]
        exp_id = exp1
        info['expert'] = exp_id

        return {'obs':obs, 'zbel':bel}, reward, done, info

    def reset(self):
        if self.reset_params:
            self.env = self.env_sampler.sample()
        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        self.last_obs = (obs, bel)
        return {'obs':obs, 'zbel':bel}


# def get_lqr_experts():
#     env = HumanoidPushingEnv()
#     estimator = ParamEnvDiscreteEstimator(env, discretization=discretization)
#     envs = estimator.envs
#     experts = []
#     for env in envs:
#         experts += [LQRControlCartPole(env)]

#     return experts

# def collect_batches(n_iterations):
#     experts = get_lqr_experts()
#     env = HumanoidPushingEnv()
#     estimator = ParamEnvDiscreteEstimator(env, discretization=discretization)
#     envs = estimator.envs

#     bayes_env = ExplicitBayesHumanoidPushingEnv(reset_params=False)
#     observations = []
#     values = []
#     new_observations = []
#     rewards = []
#     dones = []
#     actions = []
#     experiences = []

#     for env, expert in zip(envs, experts):
#         for _ in range(n_iterations):
#             done = False
#             t = 0
#             bayes_env.env = env
#             o = bayes_env.reset()
#             observations += [np.concatenate([o['obs'], o['zbel']])]
#             accum_rewards = []
#             while not done:
#                 action = expert.lqr_control(o['obs'])[0] + np.random.normal() * 0.1
#                 o, r, done, _ = bayes_env.step(action)
#                 accum_rewards += [r]

#                 if t < 300:
#                     dones += [done]
#                     rewards += [r]
#                     new_observations += [np.concatenate([o['obs'], o['zbel']])]
#                     actions += [action]
#                     observations += [np.concatenate([o['obs'], o['zbel']])]
#                 t += 1
#                 if t >= 300:
#                     break

#             value = discount(np.array(accum_rewards), 0.95)[:300]
#             values += value.tolist()

#             observations = observations[:-1]
#         experiences += [(np.array(observations), np.array(actions), np.array(values),
#                  np.array(rewards), np.array(new_observations), dones)]

#     return experiences


if __name__ == "__main__":
    # experiences = collect_batches(5)

    env = ExplicitBayesHumanoidPushingEnv()
    env.reset()

    done = False
    t = 0
    while not done:
        # action, _ = controller.lqr_control(env.env.state)
        o, r, done, _ = env.step(env.action_space.sample())
        env.render()
        # if t == 1000:
        #     break
        t += 1

    print(env.env.get_params())
    import IPython; IPython.embed()

