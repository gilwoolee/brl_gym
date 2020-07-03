from brl_gym.envs.crosswalk import CrossWalkEnv
from brl_gym.envs.crosswalk_vel import CrossWalkVelEnv
from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
import numpy as np
from scipy.stats import norm



def get_angles(poses, goals):
    poses = np.tile(poses, [1, 3]).reshape(poses.shape[0],3,-1)

    # Angles are directed straight to the goals
    diff = goals - poses
    angles = -1.0*np.arctan2(diff[:,:, 0], diff[:,:, 1])
    return angles

class BayesCrosswalkEstimator(Estimator):
    def __init__(self, env_type="velocity"):
        if env_type == "velocity":
            env = CrossWalkVelEnv()
            self.num_pedestrians = env.num_pedestrians
            goals = np.stack([env.goal_xs, np.concatenate(
                [np.arange(self.num_pedestrians // 2)+0.5, np.arange(self.num_pedestrians // 2) + 0.5])]).transpose()
            self.GOALS_RIGHT = np.tile(goals[:self.num_pedestrians // 2], [3,1]).reshape(self.num_pedestrians// 2, 3,-1)
            self.GOALS_LEFT = np.tile(goals[self.num_pedestrians // 2:],  [3,1]).reshape(self.num_pedestrians// 2, 3,-1)
        else:
            env = CrossWalkEnv()
            GOALS_RIGHT = np.tile(np.stack([[9.0, 0.5], [9.0, 1.5], [9.0, 2.5]]), [3,1]).reshape(3,3,-1)
            GOALS_LEFT  = np.tile(np.stack([[0.0, 0.5], [0.0, 1.5], [0.0, 2.5]]), [3,1]).reshape(3,3,-1)
            self.GOALS_RIGHT = GOALS_RIGHT
            self.GOALS_LEFT = GOALS_LEFT
        self.env_type = env_type
        self.num_pedestrians = env.num_pedestrians

        belief_space = Box(np.zeros((self.num_pedestrians, 3)),
                           np.ones((self.num_pedestrians, 3)), dtype=np.float32)
        self.belief_low = belief_space.low.ravel()
        self.belief_high = belief_space.high.ravel()
        super(BayesCrosswalkEstimator, self).__init__(env.observation_space, env.action_space, belief_space)

    def reset(self):
        # The goals of each pedestrian is indendent, so
        # each row tracks each pedestrian's goal belief
        self.belief = np.ones((self.num_pedestrians, 3)) / 3

        return self.get_belief()

    def estimate(self, action, observation, **kwargs):
        if action is None:
            return self.reset()
        peds, speeds, observed_angles = kwargs['pedestrians'], kwargs['pedestrian_speeds'], kwargs['pedestrian_angles']
        expected_angles = np.vstack([
                    get_angles(peds[:self.num_pedestrians // 2], self.GOALS_RIGHT),
                    get_angles(peds[self.num_pedestrians // 2:], self.GOALS_LEFT)
                    ])
        observed_angles = np.tile(observed_angles, [3,1]).transpose()
        pdf = norm.pdf(observed_angles - expected_angles, scale=0.5) * self.belief

        # posterior
        self.belief  = pdf / np.sum(pdf, axis=1).reshape(-1, 1)
        return self.get_belief()

    def get_belief(self):
        return self.belief.copy()

    def get_mle(self):
        return np.around(self.belief)

if __name__ == "__main__":
    env = CrossWalkEnv()
    obs = env.reset()
    estimator = BayesCrosswalkEstimator()
    bel = estimator.estimate(None, obs)
    print("bel", bel)

    for _ in range(100):
        action = env.action_space.sample()
        obs, r, d, info = env.step(action)
        bel = estimator.estimate(action, obs, **info)
        print("bel", np.around(bel,2))
    print(env.goals)
    import IPython; IPython.embed()

