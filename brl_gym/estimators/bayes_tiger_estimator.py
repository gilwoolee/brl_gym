import numpy as np

from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
from brl_gym.envs.tiger import Tiger, Action, Sound, TigerLocation


class BayesTigerEstimator(Estimator):
    """
    This class estimates tiger location given a known observation error
    """
    def __init__(self, obs_error=0.15,
            tiger_left_init_prob=0.5):
        env = Tiger()
        self.obs_error = obs_error
        self.tiger_left_init_prob = tiger_left_init_prob
        self.belief_high = np.ones(2)
        self.belief_low = np.zeros(2)
        self.param_high = np.ones(1)
        self.param_low = np.zeros(1)
        belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        super(BayesTigerEstimator, self).__init__(
                env.observation_space, env.action_space, belief_space)
        self.reset()

    def reset(self):
        self.belief = np.array([self.tiger_left_init_prob,
            1.0 - self.tiger_left_init_prob])
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        """
        Given the latest (action, observation), update belief
        observation: State.CLOSED_LEFT, State.CLOSED_RIGHT
        """
        if action is None or observation is Sound.SOUND_NONE:
            return self.reset()

        p_tiger_given_obs = None
        sound = kwargs['sound']

        # P(obs) = P(obs|phi_1)P(phi_1) + ... P(obs|phi_K)P(phi_K)
        if sound == Sound.SOUND_LEFT:
            p_obs = np.array([1.0 - self.obs_error, self.obs_error])
        elif sound == Sound.SOUND_RIGHT:
            p_obs = np.array([self.obs_error, 1.0 - self.obs_error])
        else:
            raise ValueError("{} unknown.".format(observation))

        if p_tiger_given_obs is None:
            p_obs_given_prior = p_obs * self.belief
            p_tiger_given_obs = p_obs_given_prior / np.sum(p_obs_given_prior)

        self.belief = p_tiger_given_obs
        return np.array([p_tiger_given_obs, 1.0 - p_tiger_given_obs])

    def get_belief(self):
        return self.belief.copy()

    def get_mle(self):
        if self.belief[0] >= 0.5:
            return np.array([TigerLocation.LEFT])
        else:
            return np.array([TigerLocation.RIGHT])


if __name__ == "__main__":
    env = Tiger()
    estimator = BayesTigerEstimator()
    state = env.reset()

    for _ in range(10):
        obs, reward, done, info = env.step(Action.LISTEN)
        print (obs)
        print (info)
        belief = estimator.estimate(Action.LISTEN, obs, **info)
        print ("belief:", estimator.get_belief())
        print ("MLE:   ", estimator.get_mle())

    # import IPython; IPython.embed()
