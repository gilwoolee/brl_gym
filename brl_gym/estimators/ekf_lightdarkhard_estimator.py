import numpy as np
from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator

class EKFLightDarkHardEstimator(Estimator):

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

    def get_belief(self):
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        if action is None:
            self.reset()
            action = np.array([0,0])

        x, cov = self.belief[:2], self.belief[-1]
        pose = self.belief[2:4]
        noise_std = observation[-1]

        action = np.clip(action, self.action_min, self.action_max)
        x_predicted = np.clip(x + action, self.x_min, self.x_max)
        cov_predicted = cov # zero process noise

        if noise_std < 0:
            # No observatrion is made, so update based on action
            self.belief = np.concatenate([x_predicted, [cov_predicted]])
            return self.get_belief()

        y = observation[:2] - x_predicted
        residual_cov = cov_predicted + noise_std ** 2
        kalman_gain = cov_predicted / residual_cov

        x_updated = x_predicted + kalman_gain * y
        cov_updated = (1 - kalman_gain) * cov_predicted

        self.belief = np.concatenate([x_updated, [cov_updated]])
        return self.get_belief()

if __name__  == "__main__":
    from brl_gym.envs.lightdarkhard import LightDarkHard
    estimator = EKFLightDarkHardEstimator()
    env = LightDarkHard()

    print ("initial belief", estimator.reset())
    action = np.array([0.5,0.0])
    for _ in range(50):
        print ("=======================")
        obs, reward, done, info = env.step(action)
        print (np.around(obs, 2))
        belief = estimator.estimate(action, obs, **info)
        print ("belief", np.around(belief,2))
        print ("true x", np.around(env.x,2))
