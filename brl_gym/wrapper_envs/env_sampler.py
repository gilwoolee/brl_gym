import numpy as np
from gym.spaces import Box

class EnvSampler:
    def sample(self):
        raise NotImplementedError

class UniformSampler:
    def sample(self, param_name, space):
        assert isinstance(space, Box)
        v = np.random.uniform(space.low, space.high)
        return v


class ResetEnvSampler:
    """
    Resets the env and returns it.
    """
    def __init__(self, env):
        self.env = env

    def sample(self):
        self.env.reset()
        return self.env


class GaussianSampler:
    def sample(self, param_name, param_space):
        # Workaround to make this function work for both of scalar and np.array
        mean = np.array((param_space.high + param_space.low) / 2.0).ravel()
        std = np.array((param_space.high - param_space.low) / 2.0).ravel()
        cov = np.diag(std ** 2)

        v = np.random.multivariate_normal(mean, cov)
        v = np.clip(v, param_space.low, param_space.high)
        return v


class ParamEnvSampler:
    """
    Sampler for Environments with param_space
    """
    def __init__(self, modifiable_env, sampler):
        self.env = modifiable_env
        self.env_cls = type(modifiable_env)
        self.param_space = self.env.param_space
        self.sampler = sampler

    def sample(self):
        env = self.env_cls()
        params = {}
        for param, space in self.param_space.items():
            params[param] = self.sampler.sample(param, space)
        env.set_params(params)
        return env


class DiscreteParamEnvSampler:
    """
    Sampler for Environments with param_space
    """
    def __init__(self, modifiable_env, n=3):
        self.env = modifiable_env
        self.env_cls = type(modifiable_env)
        self.param_space = self.env.param_space
        self.param_sampler_space = dict()
        self.n = n
        for key in self.param_space:
            self.param_sampler_space[key] = np.linspace(self.param_space[key].low[0],
                self.param_space[key].high[0], n)
            self.param_sampler_space[key] = np.around(self.param_sampler_space[key], 2)


    def sample(self):
        env = self.env_cls()
        params = {}
        for param, space in self.param_space.items():
            params[param] = np.random.choice(self.param_sampler_space[param])
        env.set_params(params)
        return env


class DiscreteUniformSampler:
    def sample(self, n):
        return np.random.choice(n)


class DiscreteEnvSampler:
    """
    Samples 1 out of the list of environments
    """
    def __init__(self, envs, discrete_sampler=DiscreteUniformSampler()):
        """
        @param envs List of envs to sample from
        @param discrete_sampler sampler over k indices
        """
        self.envs = envs
        self.sampler = discrete_sampler

    def sample(self):
        idx = self.sampler.sample(len(self.envs))
        env = self.envs[idx]
        return env
