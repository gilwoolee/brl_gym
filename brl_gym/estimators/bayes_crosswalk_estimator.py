from brl_gym.envs.crosswalk import CrossWalkEnv
from brl_gym.envs.crosswalk_vel import CrossWalkVelEnv
from gym.spaces import Box
from brl_gym.estimators.estimator import Estimator
import numpy as np
from scipy.stats import norm


def get_angles(poses, goals):
    poses = np.tile(poses, [1, goals.shape[1]]).reshape(poses.shape[0],goals.shape[1],-1)

    # Angles are directed straight to the goals
    diff = goals - poses
    angles = -1.0*np.arctan2(diff[:,:, 0], diff[:,:, 1])
    return angles


class BayesCrosswalkEstimator(Estimator):
    def __init__(self, env_type="velocity", noise_scale=0.5):
        nbins = 5
        goal_ys = np.linspace(1.5, 3.5, nbins)
        env = CrossWalkVelEnv()
        self.num_pedestrians = env.num_pedestrians
        self.GOALS_LEFT  = np.array([np.tile(env.x_left, nbins), goal_ys]).transpose()[None,:,:]
        self.GOALS_RIGHT = np.tile(np.array([np.tile(env.x_right, nbins), goal_ys]).transpose(), [2,1]).reshape(2,nbins,-1)
        self.GOALS = np.concatenate([self.GOALS_RIGHT, self.GOALS_LEFT], axis=0)
        self.num_peds_left = 2
        self.num_peds_right = 1

        self.env_type = env_type
        self.num_pedestrians = env.num_pedestrians

        # expected pose and angle concatenated
        bel_high = 4.0*np.ones((self.num_pedestrians, nbins + 3))
        bel_high[:, :nbins] = 1.0
        belief_space = Box(np.zeros((self.num_pedestrians, nbins + 3)), bel_high)
        self.belief_low = belief_space.low.ravel()
        self.belief_high = belief_space.high.ravel()
        self.noise_scale = noise_scale
        self.nbins = nbins
        self.goal_ys = goal_ys
        super(BayesCrosswalkEstimator, self).__init__(env.observation_space, env.action_space, belief_space)

    def reset(self):
        # The goals of each pedestrian is indendent, so
        # each row tracks each pedestrian's goal belief

        self.belief = np.ones((self.num_pedestrians, self.nbins)) / self.nbins

        self.belief = np.hstack([self.belief,
                                 np.mean(self.GOALS, axis=1), np.array([[-np.pi/2.0, -np.pi/2.0,np.pi/2.0]]).transpose()])

        return self.get_belief()

    def estimate(self, action, observation, **kwargs):
        if action is None:
            return self.reset()
        peds, speeds, observed_angles = kwargs['pedestrians'], kwargs['pedestrian_speeds'], kwargs['pedestrian_angles']
        expected_angles = get_angles(peds, self.GOALS)
        observed_angles = np.tile(observed_angles, [self.nbins,1]).transpose()
        pdf = norm.pdf(observed_angles - expected_angles, scale=self.noise_scale) * self.belief[:, :self.nbins]
        pdf = pdf / np.sum(pdf, axis=1).reshape(-1, 1)
        mean = np.sum(pdf[:,:,None]*self.GOALS, axis=1)
        angles = np.sum(pdf * expected_angles, axis=1)

        # posterior
        for i, speed in enumerate(speeds):
            if speed > 5e-2:
                self.belief[i, :self.nbins] = pdf[i]
                self.belief[i, self.nbins:self.nbins+2] = mean[i]
                self.belief[i, -1] = angles[i]

        return self.get_belief()

    def get_belief(self):
        return self.belief.copy()

    def get_mle(self):
        return np.around(self.belief)

if __name__ == "__main__":
    env = CrossWalkVelEnv()
    obs = env.reset()
    estimator = BayesCrosswalkEstimator("velocity")
    bel = estimator.estimate(None, obs)
    print("bel", bel)

    for _ in range(50):
        action = env.action_space.sample()
        obs, r, d, info = env.step(action)
        bel = estimator.estimate(action, obs, **info)

        print("=========")
        print(np.around(bel,2))

    print(env.goals)

