import gym

class Disturbance:
    """
    This env adds disturbance to env.
    Notably, the step function modifies the states of the env.
    but the delta-state that must be added to the nominal env.
    ResidualEnv can  access all internal variables of env.
    """
    def __init__(self, env):
        self.env = env

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError


class DisturbedEnv(gym.Env):
    """
    This env combines env and disturbance.
    """
    def __init__(self, env, disturbance, debug=False):
        self.env = env
        self.disturbance = disturbance
        self.debug = debug
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        assert hasattr(env, "_get_obs_and_reward")
        assert hasattr(env, "_get_obs")
        assert (env == disturbance.env)

    def reset(self):
        self.env.reset()
        if self.debug:
            print("obs before disturbance", self.env._get_obs())
        self.disturbance._reset()
        obs = self.env._get_obs()

        return obs

    def step(self, action):
        self.env.step(action)
        if self.debug:
            print("obs before disturbance", self.env._get_obs())

        self.disturbance._step(action)
        obs, reward, done, info = self.env._get_obs_and_reward(action)
        return obs, reward, done, info
