from itertools import product
import numpy as np
from gym.spaces import Box

from brl_gym.estimators.estimator import Estimator
from scipy.special import logsumexp

class ParamEnvDiscreteEstimator(Estimator):
    """
    Estimates parameter distribution
    """
    def __init__(self, env,
                 belief_cutoff_high=0.95,
                 belief_cutoff_low=0.05,
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
        self.noise_std = noise_std
        self.log_belief_cutoff = np.log(np.array([belief_cutoff_low, belief_cutoff_high]))
        self.belief_low = np.zeros(discretization ** num_params, dtype=np.float32)
        self.belief_high = np.ones(discretization ** num_params, dtype=np.float32)
        self.belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        self.discretization = discretization ** num_params
        self.param_values = param_values
        self.envs = envs

    def reset(self):
        self.log_belief = np.log(np.ones(self.discretization) / self.discretization)
        return np.exp(self.log_belief)

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
        if 'prev_state' not in kwargs or 'curr_state' not in kwargs:
            return self.reset()

        prev_state = kwargs['prev_state']
        curr_state = kwargs['curr_state']

        if prev_state is None:
            return self.reset()

        if np.max(self.log_belief) >= self.log_belief_cutoff[1]:
            return np.exp(self.log_belief)

        self.log_belief[self.log_belief <= self.log_belief_cutoff[0]] = 0
        self.log_belief -= logsumexp(self.log_belief)
        lp = np.zeros(len(self.log_belief))
        for i, (lb, env) in enumerate(zip(self.log_belief, self.envs)):
            if lb == 0:
                continue
            log_probability = self._estimate(env,
                                             prev_state,
                                             action,
                                             curr_state)
            lp[i] += [log_probability + lb]

        lp -= logsumexp(lp)

        self.log_belief = lp
        return np.exp(lp)


    def _estimate(self, env, prev_state, action, curr_state):
        curr_state = np.array(curr_state)
        prev_state = np.array(prev_state)
        env.reset()
        env.set_state(prev_state[:env.model.nq], prev_state[env.model.nq:])
        env.step(action)
        visited_states = env.get_state()

        # Assume N(0,noise_var)
        log_probability = -0.5*np.sum(np.linalg.norm(
            curr_state - visited_states)**2) / self.noise_std**2
        return log_probability

    def get_belief(self):
        return np.exp(self.log_belief) / np.sum(np.exp(self.log_belief))
