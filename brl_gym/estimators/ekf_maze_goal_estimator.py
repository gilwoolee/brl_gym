import numpy as np
from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
from brl_gym.envs.mujoco.maze_continuous import MazeContinuous

class EKFMazeGoalEstimator(Estimator):

    def __init__(self, init_belief=(0, 0, 1.0)):
        self.init_belief = np.array(init_belief)
        self.goal_min = np.array([-1.3, -1.3])
        self.goal_max = np.array([1.3, 1.3])

        self.action_min = -1
        self.action_max = 1
        self.belief_low =  np.concatenate([self.goal_min, [0]])
        self.belief_high = np.concatenate([self.goal_max, [2.0]])
        env = MazeContinuous()
        belief_space = Box(self.belief_low, self.belief_high, dtype=np.float32)
        super(EKFMazeGoalEstimator, self).__init__(
                env.observation_space, env.action_space, belief_space)

        self.param_low = self.goal_min
        self.param_high = self.goal_max

        self.reset()

    def reset(self):
        self.belief = self.init_belief.copy()

    def get_belief(self):
        return self.belief.copy()

    def estimate(self, action, observation, **kwargs):
        agent_pos = observation[:2]
        dist_to_goal_observed = observation[-2:]

        if np.all(dist_to_goal_observed == np.array([0,0])):
            # No update
            return self.get_belief()

        goal_previous, cov = self.belief[:2], self.belief[-1]
        dist_to_goal_predicted = goal_previous - agent_pos

        noise_std = kwargs['noise_scale']

        if action is None:
            self.reset()

        cov_predicted = cov # zero process noise

        y = dist_to_goal_observed - dist_to_goal_predicted
        residual_cov = cov_predicted + noise_std ** 2
        kalman_gain = cov_predicted / residual_cov

        dist_to_goal_updated = dist_to_goal_predicted + kalman_gain * y

        goal_updated = agent_pos + dist_to_goal_updated
        cov_updated = (1 - kalman_gain) * cov_predicted

        self.belief = np.concatenate([goal_updated, [cov_updated]])
        return self.get_belief()

    def get_mle(self):
        return self.get_belief().copy()[:2]

if __name__  == "__main__":
    from brl_gym.envs.mujoco.maze_continuous import MazeContinuous
    estimator = EKFMazeGoalEstimator()
    env = MazeContinuous()

    print ("initial belief", estimator.reset())
    action = np.array([1.0,1.0,1.0])
    for _ in range(100):
        print ("=======================")
        obs, reward, done, info = env.step(action)
        print (obs, reward, done, info)
        belief = estimator.estimate(action, obs, **info)
        print ("belief", np.around(belief,2))
        print ("true goal", env.model.site_pos[env.target_sid][:2])


