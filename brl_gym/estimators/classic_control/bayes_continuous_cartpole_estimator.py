from brl_gym.estimators.param_env_discrete_estimator import ParamEnvDiscreteEstimator

from gym.spaces import Box
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv, NoisyContinuousCartPoleEnv
from scipy.stats import norm
import numpy as np

class BayesContinuousCartpoleEstimator(ParamEnvDiscreteEstimator):
    def __init__(self, noise=False):
        if not noise:
            env = ContinuousCartPoleEnv(ctrl_noise_scale=0.0)
        else:
            env = NoisyContinuousCartPoleEnv(ctrl_noise_scale=0.0)
        super(BayesContinuousCartpoleEstimator, self).__init__(env, discretization=5,
            belief_cutoff_low=0.05)
        for env in self.envs:
            assert env.ctrl_noise_scale == 0.0

    def estimate(self, action, observation, **kwargs):
        if action == None:
            self.prev_state = observation.copy()
            return self.reset()

        kwargs['prev_state'] = self.prev_state.copy()
        kwargs['curr_state'] = observation.copy()
        bel = super().estimate(action, observation, **kwargs)
        self.prev_state = observation.copy()

        return bel

    def _estimate(self, env, prev_state, action, curr_state):
        env.set_state(prev_state)
        visited_state, _, d, _ = env.step(action)
        log_probability = np.sum(norm.logpdf(
                            curr_state - visited_state, scale=self.noise_std))
        if d:
            env.reset()
        return log_probability

if __name__ == "__main__":
    env = ContinuousCartPoleEnv()
    estimator = BayesContinuousCartpoleEstimator()
    obs = env.reset()
    print("length", env.length, 'masscart', env.masscart)

    bel = estimator.estimate(None, obs)
    print("bel", bel)
    for _ in range(200):
        action = env.action_space.sample()
        obs, _, d, info  = env.step(action)
        bel = estimator.estimate(action, obs, **info)
        if d:
            break
        print(np.around(bel, 2))


