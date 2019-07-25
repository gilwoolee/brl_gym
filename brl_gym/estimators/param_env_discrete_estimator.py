from itertools import product
import numpy as np
from gym.spaces import Box

from brl_gym.estimators.estimator import Estimator
from scipy.misc import logsumexp

class ParamEnvDiscreteEstimator(Estimator):
    """
    Estimates parameter distribution
    """
    def __init__(self, env,
                 belief_cutoff=0.7,
                 noise_std=0.5,
                 discretization=5):
        """
        @param discretization Number of discretization for each space in param_space
        """
        self.env_class = type(env)
        param_space = env.param_space
        num_params = len(param_space.keys())

        param_values = []
        for name, space in sorted(param_space.items()):

            dimension = space.shape
            if len(dimension) == 0:
                dimension = 1
            else:
                dimension = dimension[0]
            param_range = np.zeros((discretization, dimension))

            # For each parameter, linspace is applied together
            if dimension == 1:
                param_range[:, 0] = np.linspace(space.low[0], space.high[0], discretization)
            else:
                for dim in range(dimension):
                    param_range[:, dim] = np.linspace(space.low[dim], space.high[dim], discretization)
            param_values.append(param_range.tolist())

        envs = []
        for values in product(*param_values):
            params = dict()
            for name, v in zip(sorted(param_space.keys()), values):
                params[name] = np.array(v)
            env = self.env_class()
            env.set_params(params)
            envs += [env]

        num_params = len(param_space.keys())
        self.param_space = param_space
        self.noise_var = noise_std**2
        self.log_belief_cutoff = np.log(belief_cutoff)
        self.belief_low = np.zeros(discretization ** num_params, dtype=np.float32)
        self.belief_high = np.ones(discretization ** num_params, dtype=np.float32)
        self.belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        self.discretization = discretization ** num_params
        self.param_values = param_values
        self.envs = envs

        self.reset()

    def reset(self):
        self.log_belief = np.log(np.ones(self.discretization) / self.discretization)

    def get_best_env(self):
        """
        Returns the most likely environment given the current estimate
        """
        idx = np.argmax(self.log_belief)
        env = self.envs[idx]
        return env

    def get_best_params(self):
        """
        Returne the most likely params given the current estimate
        """
        params = self.get_best_env().get_params()
        params_copy = dict()
        for name in self.param_space:
            params_copy[name] = params[name]

        return params_copy

    def estimate(self, action, observation, **kwargs):
        # import IPython; IPython.embed(); import sys; sys.exit(0)
        if 'prev_state' not in kwargs or 'curr_state' not in kwargs:
            self.reset()
            return self.log_belief

        prev_state = kwargs['prev_state']
        curr_state = kwargs['curr_state']

        if prev_state is None:
            self.reset()
            return self.log_belief

        if np.max(self.log_belief) >= self.log_belief_cutoff:
            return self.log_belief

        log_probabilities = []
        for b, env in zip(self.log_belief, self.envs):
            log_probability = self._estimate(env,
                                             prev_state,
                                             action,
                                             curr_state)
            log_probability += b
            log_probabilities += [log_probability]

        log_probabiliites = np.array(log_probabilities)
        log_probabilities -= logsumexp(log_probabilities)

        self.log_belief = log_probabilities
        return log_probabiliites


    def _estimate(self, env, prev_state, action, curr_state):
        curr_state = np.array(curr_state)
        prev_state = np.array(prev_state)
        env.reset()
        env.set_state(prev_state[:env.model.nq], prev_state[env.model.nq:])
        env.step(action)
        visited_states = env.get_state()

        # Assume N(0,noise_var)
        log_probability = -0.5*np.sum(np.linalg.norm(
            curr_state - visited_states)**2) / self.noise_var
        return log_probability

    def get_belief(self):
        return np.exp(self.log_belief) / np.sum(np.exp(self.log_belief))
