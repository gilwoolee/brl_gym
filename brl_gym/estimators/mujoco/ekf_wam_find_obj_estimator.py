import numpy as np
from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator

class EKFWamFindObjEstimator(Estimator):
    """
    EKF to track the object on the shelf
    """

    def __init__(self, action_space, std_range=[0.1, 2.0]):
        self.pos_min = np.array([0, -0.7, 0.05])
        self.pos_max = np.array([0.2, 0.1, 0.5])
        self.init_belief = np.array([0.1, 0.3, 0.2, 0.5])

        # Tracks object location and std
        self.belief_low =  np.concatenate([self.pos_min, [std_range[0]]])
        self.belief_high = np.concatenate([self.pos_max, [std_range[1]]])

        self.param_space = Box(self.pos_min, self.pos_max, dtype='float32')
        self.belief_space = Box(self.belief_low, self.belief_high, dtype='float32')

        self.action_space = action_space

    def reset(self):
        self.belief = self.init_belief.copy()
        return self.belief.copy()

    def get_belief_for_dist_to_obj(self, belief, grip_pos):
        # Converts obj position to dist to obj
        dist_to_obj = belief[:3] - grip_pos
        belief = np.concatenate([dist_to_obj, belief[-1:]])
        return belief

    def get_belief(self):
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        if action is None:
            return self.reset()

        grip_pos, noisy_dist, noise_std = observation[:3], observation[3:6], observation[-1]
        noisy_observed_obj_pos = grip_pos + noisy_dist

        prev_obj_pos, prev_std = self.belief[:self.pos_min.shape[0]], self.belief[-1]

        noise_std = observation[-1]

        # Object doesn't move
        obj_predicted = prev_obj_pos
        std_predicted = prev_std # zero process noise

        y = noisy_observed_obj_pos - obj_predicted
        residual_cov = std_predicted**2 + noise_std ** 2
        kalman_gain = std_predicted**2 / residual_cov

        x_updated = obj_predicted + kalman_gain * y
        cov_updated = (1 - kalman_gain) * std_predicted**2

        self.belief = np.concatenate([x_updated, [np.sqrt(cov_updated)]])
        return self.get_belief_for_dist_to_obj(self.belief, grip_pos)


if __name__  == "__main__":
    from brl_gym.envs.mujoco.wam_find_obj import WamFindObjEnv
    env = WamFindObjEnv()
    estimator = EKFWamFindObjEstimator(env.action_space)

    print ("initial belief", estimator.reset())
    action = np.array([1.0,1.0,1.0])
    for _ in range(100):
        print ("=======================")
        obs, reward, done, info = env.step(action)
        belief = estimator.estimate(action, obs, **info)
        print ("belief          ", np.around(belief,1))
        print ("true dist-to-obj", env.obj_pos - env.grip_pos)
        print ("obj pos         ", env.obj_pos)
        print ("belief-obj      ", np.around(estimator.belief,2))


