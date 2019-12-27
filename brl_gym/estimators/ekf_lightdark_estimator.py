import numpy as np
from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator

class EKFLightDarkEstimator(Estimator):

    def __init__(self, init_belief=(2, 2, 2.25), x_min=(-1, -2), x_max=(7, 4),
            action_min=-0.5, action_max=0.5):
        self.init_belief = np.array(init_belief)
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
        self.action_min = action_min
        self.action_max = action_max
        self.belief_low =  np.concatenate([self.x_min, [0]])
        self.belief_high = np.concatenate([self.x_max, [5]])

        self.param_low = self.x_min
        self.param_high = self.x_max

        self.belief_space = Box(self.belief_low, self.belief_high, dtype='float32')
        self.reset()

    def reset(self):
        self.belief = self.init_belief.copy()

    def get_belief_for_dist_to_goal(self, belief, goal):
        dist_to_goal = belief[:2] - goal
        belief = np.concatenate([dist_to_goal, belief[-1:]])
        return belief

    def get_belief(self):
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        x, cov = self.belief[:2], self.belief[-1]
        goal = self.belief[2:4]
        self.goal = goal
        noise_std = observation[-1]

        if action is None:
            self.reset()
            action = np.array([0,0])

        assert(self.action_min == -0.5)
        action = np.clip(action, self.action_min, self.action_max)
        x_predicted = np.clip(x + action, self.x_min, self.x_max)
        cov_predicted = cov # zero process noise

        y = observation[:2] - x_predicted
        residual_cov = cov_predicted + noise_std ** 2
        kalman_gain = cov_predicted / residual_cov

        x_updated = x_predicted + kalman_gain * y
        cov_updated = (1 - kalman_gain) * cov_predicted

        self.belief = np.concatenate([x_updated, [cov_updated]])
        return self.get_belief_for_dist_to_goal(self.belief, goal)

    def get_mle(self):
        belief = self.get_belief_for_dist_to_goal(self.belief, self.goal)
        return belief[:2]

if __name__  == "__main__":
    from bpo_gym.envs.lightdark import LightDark
    estimator = EKFLightDarkEstimator()
    env = LightDark()

    print ("initial belief", estimator.reset())
    action = np.array([1.0,1.0])
    for _ in range(5):
        print ("=======================")
        obs, reward, done, info = env.step(action)
        print (obs, reward, done, info)
        belief = estimator.estimate(action, obs, **info)
        print ("belief", belief)
        print ("true dist-to-goal", env.goal - env.x)


