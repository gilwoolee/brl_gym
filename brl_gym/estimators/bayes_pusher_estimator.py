import numpy as np

from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
from brl_gym.envs.mujoco.pusher import Pusher, TARGET_LOCATIONS
from scipy.stats import norm


class BayesPusherEstimator(Estimator):
    """
    This class estimates tiger location given a known observation error
    """
    def __init__(self):
        env = Pusher()
        self.belief_high = np.ones(5)
        self.belief_low = np.zeros(5)
        belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        super(BayesPusherEstimator, self).__init__(
                env.observation_space, env.action_space, belief_space)
        self.reset()

    def reset(self):
        self.belief = np.ones(TARGET_LOCATIONS.shape[0])
        self.belief /= np.sum(self.belief)
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        """
        Given the latest (action, observation), update belief
        observation: State.CLOSED_LEFT, State.CLOSED_RIGHT
        """
        if action is None:
            return self.reset()

        # obs = observation[3:5]
        obs_goal_dist = kwargs['goal_dist']
        dist_to_goals = np.linalg.norm(observation[12:].reshape(-1,2), axis=1)

        # dist_to_up = np.abs(obs - self.target_locations[0,1])
        # dist_to_down = np.abs(obs - self.target_locations[1,1])

        p_obs_given_prior = norm.pdf(dist_to_goals - obs_goal_dist, scale=1.0) * self.belief
        p_goal_obs  = p_obs_given_prior / np.sum(p_obs_given_prior)
        # p_obs_given_up = norm.pdf(dist_to_up - goal_dist, scale=1.0) * self.belief[0]
        # p_obs_given_down = norm.pdf(dist_to_down - goal_dist, scale=1.0) * self.belief[1]
        # p_obs_given_prior = p_obs_given_up + p_obs_given_down
        # p_up_given_obs = p_obs_given_up / p_obs_given_prior
        # self.belief = np.array([p_up_given_obs, 1 - p_up_given_obs])
        self.belief = p_goal_obs
        return self.belief

    def get_belief(self):
        return self.belief.copy()


if __name__ == "__main__":
    env = Pusher()
    estimator = BayesPusherEstimator()
    state = env.reset()

    for _ in range(50):
        a = np.array([1.0, 0.0, 0.0])
        obs, reward, done, info = env.step(a)
        print (np.around(obs[:2],2), info)
        belief = estimator.estimate(a, obs, **info)
        print ("belief:", estimator.get_belief())


    import IPython; IPython.embed()
