import numpy as np

import gym
from gym.spaces import Box, Discrete

from brl_gym.wrapper_envs import WrapperEnv
from brl_gym.wrapper_envs.util import to_one_hot


class MLEEnv(WrapperEnv):
    """
    An environment with latent MDP estimator and a modifiable env.
    Best estimate is augmented to observation.
    """
    def __init__(self, env, estimator, augment_to_obs=True):

        super(MLEEnv, self).__init__(env, estimator, augment_to_obs)

        # Augment with mle space
        if isinstance(env.observation_space, Box):
            lows = np.concatenate([
                env.observation_space.low,
                estimator.param_low], axis=0)
            highs = np.concatenate([
                env.observation_space.high,
                estimator.param_high], axis=0)

        else:
            lows = np.concatenate([
                np.zeros(env.observation_space.n),
                estimator.param_low], axis=0)
            highs = np.concatenate([
                np.ones(env.observation_space.n),
                estimator.param_high], axis=0)

        self.observation_space = Box(lows, highs, dtype=np.float32)

    def _augment_observation(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        self.estimator.estimate(
                action, obs, **kwargs)
        mle = self.estimator.get_mle()

        if not isinstance(mle, np.ndarray):
            mle = np.array([mle])

        if isinstance(self.env.observation_space, Discrete):
            obs = to_one_hot(obs, self.env.observation_space.n)

        if not isinstance(obs, np.ndarray):
            obs = np.array([obs])

        kwargs['mle'] = mle
        if self.augment_to_obs:
            return np.concatenate([obs, mle], axis=0), kwargs
        else:

            return obs, kwargs
