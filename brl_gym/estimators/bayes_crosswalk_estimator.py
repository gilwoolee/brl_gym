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
    def __init__(self, noise_scale=0.5):
        nbins = 5
        env = CrossWalkVelEnv()
        goal_ys = self.goal_ys = np.linspace(env.y_starts[0], env.y_starts[1], nbins)
        self.num_pedestrians = env.num_pedestrians
        self.env = env

        # expected pose and angle concatenated
        bel_high = 4.0*np.ones((self.num_pedestrians, nbins + 3))
        bel_high[:, :nbins] = 1.0
        belief_space = Box(np.zeros((self.num_pedestrians, nbins + 3)), bel_high)
        self.belief_low = belief_space.low.ravel()
        self.belief_high = belief_space.high.ravel()
        self.noise_scale = noise_scale
        self.nbins = nbins
        self.GOALS_LEFT  = np.array([np.tile(self.env.x_left,  nbins), goal_ys]).transpose().reshape(1, nbins, -1)
        self.GOALS_RIGHT = np.array([np.tile(self.env.x_right, nbins), goal_ys]).transpose().reshape(1, nbins,-1)
        super(BayesCrosswalkEstimator, self).__init__(env.observation_space, env.action_space, belief_space)

    def reset(self, obs=None, **kwargs):
        if obs is None:
            return self.belief_low
        peds = kwargs['pedestrians']
        lefts = peds[:,0] < 0.5
        goal_ys, nbins = self.goal_ys, self.nbins

        self.GOALS = np.zeros((3,nbins,2), np.float32)
        self.GOALS[lefts] = self.GOALS_RIGHT
        self.GOALS[np.logical_not(lefts)] = self.GOALS_LEFT

        # The goals of each pedestrian is indendent, so
        # each row tracks each pedestrian's goal belief
        self.belief = np.ones((self.num_pedestrians, self.nbins)) / self.nbins

        angles = np.pi/2.0 * np.ones((3,1))
        angles[peds[:,0] < 0.5] *= -1.0
        self.belief = np.hstack([self.belief,
                                 np.mean(self.GOALS, axis=1), angles])

        return self.get_belief()

    def estimate(self, action, observation, **kwargs):
        if action is None:
            return self.reset(observation, **kwargs)
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
    estimator = BayesCrosswalkEstimator()
    peds = obs[8:14].reshape(-1, 2)
    bel = estimator.estimate(None, obs, pedestrians=peds)
    print("peds")
    print(peds)
    print("==========")
    print(np.around(bel,2))

    for _ in range(20):
        action = env.action_space.sample()
        obs, r, d, info = env.step(action)
        bel = estimator.estimate(action, obs, **info)

        print("=========")
        print(np.around(bel,2))

    print(env.goals)

