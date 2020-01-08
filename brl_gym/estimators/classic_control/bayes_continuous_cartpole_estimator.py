from brl_gym.estimators.estimator import Estimator
from gym.spaces import Box
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv
from scipy.stats import norm
import numpy as np

class BayesContinuousCartpoleEstimator(Estimator):
    def __init__(self, param_range=np.linspace(0.5, 2.0, 3)):
        self.param_range = param_range

        self.envs = [ContinuousCartPoleEnv(ctrl_noise_scale=0.0, random_param=False)
                     for _ in range(len(self.param_range))]
        for l, env in zip(self.param_range, self.envs):
            env.set_params(dict(length=l))

        self.belief_low = np.zeros(len(self.param_range), dtype=np.float32)
        self.belief_high = np.ones(len(self.param_range), dtype=np.float32)

        self.belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        self.param_space = Box(np.array([param_range[0]]), np.array([param_range[1]]))

        env = self.envs[0]
        super(BayesContinuousCartpoleEstimator, self).__init__(
                env.observation_space, env.action_space, self.belief_space)

        # Observation noise (if zero, it means that the cartpole is perfect
        # and hence the belief collapse to 1 after one step.)
        self.noise_scale = 0.5

    def reset(self):
        self.belief = np.ones(len(self.param_range)) / len(self.param_range)
        return self.belief.ravel()

    def estimate(self, action, observation, **kwargs):
        if action == None:
            self.prev_state = observation.copy()
            return self.reset()
        probs = np.zeros(len(self.param_range))
        for i, env in enumerate(self.envs):
            env.set_state(self.prev_state)
            state, _, d, _ = env.step(action)
            if d:
                env.reset()
            probs[i] = np.exp(np.sum(norm.logpdf(observation - state, scale=self.noise_scale)))
        probs *= self.belief
        self.belief = probs / np.sum(probs)
        self.prev_state = observation
        return self.belief.copy()

        raise NotImplementedError

    def get_belief(self):
        return self.belief.copy().ravel()

    def get_mle(self):
        best_param = self.param_range[np.argmax(self.belief)]
        return best_param


if __name__ == "__main__":
    env = ContinuousCartPoleEnv(ctrl_noise_scale=1.0, random_param=True)
    estimator = BayesContinuousCartpoleEstimator()
    obs = env.reset()
    print("length", env.length)
    bel = estimator.estimate(None, obs)
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, _, info  = env.step(action)
        bel = estimator.estimate(action, obs, **info)
        print(np.around(bel, 1))


