from brl_gym.estimators.estimator import Estimator
from gym.spaces import Box
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv
from scipy.stats import norm

class BayesContinuousCartpoleEstimator(Estimator):
    def __init__(self):
        # Length of the pole, range [0.5, 0.6, ... 1.5]
        self.param_range = np.linspace(0.5, 1.5, 11)

        self.envs = [ContinuousCartPoleEnv(ctrl_noise_scale=0.0) \
                     for _ in range(len(self.param_range)]
        for l, env in zip(self.param_range, self.envs):
            env.set_params(dict(length=l))

        self.belief_low = np.zeros(len(self.param_range), dtype=np.float32)
        self.belief_high = np.ones(len(self.param_range), dtype=np.float32)

        self.belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        self.param_space = Box(np.array([0.5]), np.array(1.5))

        env = self.envs[0]
        super(BayesContinuousCartpoleEstimator, self).__init__(
                env.observation_space, env.action_space, self.belief_space))

        self.seed()

        # Observation noise (if zero, it means that the cartpole is perfect
        # and hence the belief collapse to 1 after one step.
        self.noise_scale = 0.5

    def reset(self):
        self.belief = np.ones(len(self.param_range))
        return self.belief.ravel()

    def estimate(self, action, observation, **kwargs):
        probs = np.zeros(len(self.param_range))
        for i, env in enumerate(self.envs):
            env.set_state(observation)
            state, _, _, _ = env.step(action)
            probs[i] = norm.pdf(observation - state, scale=self.noise_scale)
        probs *= self.belief
        self.belief = probs / np.sum(probs)
        return self.belief.copy()

        raise NotImplementedError

    def get_belief(self):
        return self.belief.copy().ravel()

    def get_mle(self):
        best_param = self.param_range[np.argmax(self.belief)]
        return best_param




