import numpy as np
from brl_gym.estimators.bayes_doors_estimator import BayesDoorsEstimator #, LearnableDoorsBF
from brl_gym.envs.mujoco.doors import DoorsEnv, NoisyDoorEnv
from brl_gym.envs.mujoco.doors_slow import DoorsSlowEnv
from brl_gym.wrapper_envs import BayesEnv
from brl_gym.wrapper_envs.env_sampler import DiscreteEnvSampler

from gym.spaces import Box, Dict
from gym import utils

class ExplicitBayesDoorsEnv(BayesEnv, utils.EzPickle):
    def __init__(self, reset_params=True,
        reward_entropy=False, entropy_weight=0.0,
        doors_slow=False, disturbance=False,
        estimate_disturbance=False, learned_residual=None):
        assert not (disturbance and learned_residual is not None)

        self.num_doors = 4
        self.num_cases = 2**self.num_doors
        self.cases =  ['{{:0{}b}}'.format(self.num_doors).format(x) \
                      for x in range(self.num_cases)]
        self.cases_np = [np.array([int(x) for x in case]) for case in self.cases]

        envs = []

        env_class = DoorsEnv if not doors_slow else DoorsSlowEnv
        if disturbance:
            env_class = NoisyDoorEnv

        for case in self.cases_np:
            if not disturbance:
                env = env_class(disturbance=learned_residual)
            else:
                env = env_class()
            env.open_doors = case.astype(np.bool)
            envs += [env]

        self.env_sampler = DiscreteEnvSampler(envs)
        self.estimator = BayesDoorsEstimator(
            estimate_disturbance=estimate_disturbance, residual=learned_residual)

        super(ExplicitBayesDoorsEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        obs_space = Box(low=np.concatenate([env.observation_space.low, self.estimator.param_space.low]),
                        high=np.concatenate([env.observation_space.high, self.estimator.param_space.high]))
        self.observation_space = obs_space
        # Dict(
        #     {"obs": env.observation_space, "zbel": self.estimator.belief_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params
        self.reward_entropy = reward_entropy
        if reward_entropy:
            self.entropy_weight = entropy_weight
        else:
            self.entropy_weight = 0.0
        utils.EzPickle.__init__(self)

    def _update_belief(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        self.estimator.estimate(
                action, obs, **kwargs)
        belief = self.estimator.get_belief()
        return belief, kwargs

    def get_flat_belief():
        return self.estimator.get_flat_belief()

    def step(self, action):
        prev_state = self.env.get_state().copy()
        obs, reward, done, info = self.env.step(action)
        info['prev_state'] = prev_state
        info['curr_state'] = self.env.get_state()
        info['done'] = done

        bel, info = self._update_belief(
                                        action,
                                        obs,
                                        **info)
        entropy = np.sum(-np.log(bel+1e-5)/np.log(len(bel)) * (bel + 1e-5))
        ent_reward = -(entropy - self.prev_entropy)
        self.prev_entropy = entropy
        # reward += ent_reward * self.entropy_weight
        info['entropy'] = entropy
        # self.color_belief()
        info['label'] = self.nominal_env.open_doors.astype(np.int)
        obs = np.concatenate([obs, bel])
        return obs, reward, done, info

    def reset(self):
        if self.reset_params:
            while True:
                self.env = self.env_sampler.sample()
                if not np.all(self.env.open_doors == False):
                    break

        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        entropy = np.sum(-np.log(bel)/np.log(bel.shape[0]) * bel)
        self.prev_entropy = entropy
        obs = np.concatenate([obs, bel])

        return obs


    def color_belief(self):
        bel = self.estimator.belief
        for i, b in enumerate(bel.ravel()):
            self.env.model.geom_rgba[10+i, -1] = 1 - b

    def set_bayes_filter(self, file):
        self.estimator.set_bayes_filter(file)

class ExplicitBayesDoorsEnvNoEntropyReward(ExplicitBayesDoorsEnv):
    def __init__(self):
        super(ExplicitBayesDoorsEnvNoEntropyReward, self).__init__(True, False)


# Instead of the belief, return best estimate
class UPMLEDoorsEnv(BayesEnv, utils.EzPickle):
    def __init__(self, reset_params=True, reward_entropy=True):

        self.num_doors = 4
        self.num_cases = 2**self.num_doors
        self.cases =  ['{{:0{}b}}'.format(self.num_doors).format(x) \
                      for x in range(self.num_cases)]
        self.cases_np = [np.array([int(x) for x in case]) for case in self.cases]

        envs = []
        for case in self.cases_np:
            env = DoorsEnv()
            env.open_doors = case.astype(np.bool)
            envs += [env]

        self.estimator = BayesDoorsEstimator()

        self.env_sampler = DiscreteEnvSampler(envs)
        super(UPMLEDoorsEnv, self).__init__(env, self.estimator)
        self.nominal_env = env

        self.observation_space = Dict(
            {"obs": env.observation_space, "zparam": self.estimator.param_space})
        self.internal_observation_space = env.observation_space
        self.env = env
        self.reset_params = reset_params
        self.reward_entropy = reward_entropy
        utils.EzPickle.__init__(self)

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
        prev_state = self.env.get_state().copy()
        obs, reward, done, info = self.env.step(action)
        info['prev_state'] = prev_state
        info['curr_state'] = self.env.get_state()

        bel, info = self._update_belief(
                                        action,
                                        obs,
                                        **info)
        entropy = np.sum(-np.log(bel+1e-5)/np.log(len(bel)) * (bel + 1e-5))
        ent_reward = -(entropy - self.prev_entropy)
        self.prev_entropy = entropy
        if self.reward_entropy:
            reward += ent_reward
        info['entropy'] = entropy
        param = self.estimator.get_mle()

        return {'obs':obs, 'zparam':param}, reward, done, info

    def reset(self):
        if self.reset_params:
            while True:
                self.env = self.env_sampler.sample()
                if not np.all(self.env.open_doors == False):
                    break

        obs = self.env.reset()
        self.estimator.reset()
        bel, _ = self._update_belief(action=None, obs=obs)
        entropy = np.sum(-np.log(bel)/np.log(bel.shape[0]) * bel)
        self.prev_entropy = entropy
        param = self.estimator.get_mle()
        return {'obs':obs, 'zparam':param}


class UPMLEDoorsEnvNoEntropyReward(UPMLEDoorsEnv):
    def __init__(self):
        super(UPMLEDoorsEnvNoEntropyReward, self).__init__(True, False)


class BayesDoorsEntropyEnv(ExplicitBayesDoorsEnv):
    """
    Environment that provides entropy instead of belief as observation
    """
    def __init__(self, reward_entropy=True, reset_params=True, observe_entropy=True):
        super(BayesDoorsEntropyEnv, self).__init__(reward_entropy=reward_entropy, reset_params=reset_params)
        utils.EzPickle.__init__(self)

        entropy_space = Box(np.array([0.0]), np.array([1.0]))
        if observe_entropy:
            self.observation_space = Dict(
                {"obs": env.observation_space, "zentropy": entropy_space})
        else:
            self.observation_space = env.observation_space

        self.observe_entropy = observe_entropy

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['bel'] = obs['zbel'].copy()

        del obs['zbel']
        if self.observe_entropy:
            obs['zentropy'] = np.array([info['entropy']])
            return obs, reward, done, info
        else:
            return obs['obs'], reward, done, info

    def reset(self):
        obs = super().reset()
        if self.observe_entropy:
            obs['zentropy'] = np.array([self.prev_entropy])
            return obs
        else:
            return obs['obs']

if __name__ == "__main__":
    env = ExplicitBayesDoorsEnv(disturbance=True, estimate_disturbance=True, entropy_weight=0.0)
    obs = env.reset()
    doors = env.env.open_doors

    done = False
    rewards = []
    while not done:
        env.step(env.action_space.sample())
        env.render()

    # Test upmle env
    # env = UPMLEDoorsEnv()
    # obs = env.reset()
    # for _ in range(100):
    #     o, r, d, info = env.step([0,0,1])
    #     print(o['zparam'], env.estimator.belief)
    # import IPython; IPython.embed()

    # Test entropy-only env
    from brl_gym.experts.doors.expert import DoorsExpert as Expert

    env = BayesDoorsEntropyEnv()
    expert = Expert(mle=True)
    o = env.reset()
    print(o)
    info = []

    for _ in range(300):
        o = np.concatenate([o['obs'], o['zentropy']], axis=0).reshape(1, -1)
        action = expert.action(o, info).ravel()
        action[-1] = 1
        o, r, d, info = env.step(action)
        info = [info]
        if d:
            break
        env.render()
        print("expert action", action, np.around(info[0]['bel'],1))
    import IPython; IPython.embed()
