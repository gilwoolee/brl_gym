import numpy as np
from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
from brl_gym.envs.lightdarkhard import LightDarkHard
class EKFLightDarkHardEstimator(Estimator):

    def __init__(self, init_belief=(0, 0, 4.0)):
        self.init_belief = np.array(init_belief)
        env = LightDarkHard()
        self.x_min = env.pos_min
        self.x_max = env.pos_max

        self.action_min = env.action_space.low
        self.action_max = env.action_space.high

        self.belief_low =  np.concatenate([self.x_min, [0]])
        self.belief_high = np.concatenate([self.x_max, [4.0]])

        self.param_low = self.x_min
        self.param_high = self.x_max

        self.belief_space = Box(self.belief_low, self.belief_high, dtype='float32')
        self.reset()

    def reset(self):
        self.belief = self.init_belief.copy()
        return self.get_belief()

    def get_belief(self):
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        if action is None:
            self.reset()
            action = np.array([0,0])

        x, cov = self.belief[:2], self.belief[-1] ** 2
        noise_std = observation[-1]
        action = np.clip(action, self.action_min, self.action_max)
        x_predicted = np.clip(x + action, self.x_min, self.x_max)
        cov_predicted = cov # zero process noise

        if noise_std < 0:
            # No observatrion is made, so update based on action
            self.belief = np.concatenate([x_predicted, self.belief[-1:]])
            return self.get_belief()

        y = observation[:2] - x_predicted
        residual_cov = cov_predicted + noise_std ** 2
        kalman_gain = cov_predicted / residual_cov

        x_updated = x_predicted + kalman_gain * y
        cov_updated = (1 - kalman_gain) * cov_predicted
        self.belief = np.concatenate([x_updated, [np.sqrt(cov_updated)]])
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
        print ("obs   ", np.around(obs, 2))
        belief = estimator.estimate(action, obs, **info)
        print ("belief", np.around(belief,2))
        print ("true x", np.around(env.x,2))

