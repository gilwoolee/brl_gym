import numpy as np

from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
from brl_gym.envs.mujoco.point_mass import PointMassEnv, GOAL_POSE
from scipy.stats import norm


class BayesMazeEstimator(Estimator):
    """
    This class estimates tiger location given a known observation error
    """
    def __init__(self):
        env = PointMassEnv()
        self.belief_high = np.ones(GOAL_POSE.shape[0])
        self.belief_low = np.zeros(GOAL_POSE.shape[0])
        belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        super(BayesMazeEstimator, self).__init__(
                env.observation_space, env.action_space, belief_space)
        self.reset()

    def reset(self):
        self.belief = np.ones(GOAL_POSE.shape[0])
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
        if 'goal_dist' in kwargs:
            obs_goal_dist = kwargs['goal_dist']
            dist_to_goals = np.linalg.norm(observation[4:4+GOAL_POSE.shape[0]*2].reshape(-1,2), axis=1)
            p_obs_given_prior = norm.pdf(dist_to_goals - obs_goal_dist, scale=1.0) * self.belief
            p_goal_obs  = p_obs_given_prior / np.sum(p_obs_given_prior)
            self.belief = p_goal_obs
            return self.belief
        else:
            return self.belief

    def get_belief(self):
        return self.belief.copy()


if __name__ == "__main__":
    env = PointMassEnv()
    estimator = BayesMazeEstimator()
    state = env.reset()

    for _ in range(50):
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)

        belief = estimator.estimate(a, obs, **info)
        print ("belief:", np.around(estimator.get_belief(),1))


    import IPython; IPython.embed()
