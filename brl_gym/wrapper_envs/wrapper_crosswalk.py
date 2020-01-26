import numpy as np

from brl_gym.wrapper_envs import BayesEnv
from brl_gym.envs.crosswalk import CrossWalkEnv
from brl_gym.estimators.bayes_crosswalk_estimator import BayesCrosswalkEstimator, GOALS_LEFT, GOALS_RIGHT, get_angles

def get_pedestrian_directions(speeds, angles):
    return speeds.reshape(-1,1) \
            * np.array([-np.sin(angles),
                    np.cos(angles)]).transpose()

class BayesCrossWalkEnv(BayesEnv):
    # Wrapper envs for mujoco envs
    def __init__(self):
        self.env = CrossWalkEnv()
        self.estimator = BayesCrosswalkEstimator()
        self.num_pedestrians = self.env.num_pedestrians
        super(BayesCrossWalkEnv, self).__init__(self.env, self.estimator)

    def _expand_belief(self, obs, belief, **kwargs):
        # Convert belief into useful features

        if 'pedestrians' not in kwargs:
            peds = obs[-self.num_pedestrians*2*2:-self.num_pedestrians*2].reshape(-1, 2)
            directions = obs[-self.num_pedestrians*2:].reshape(-1,2) - peds
            speeds = np.linalg.norm(directions, axis=1)
        else:
            peds, speeds = kwargs['pedestrians'], kwargs['pedestrian_speeds']
        expected_angles = np.vstack([
                    get_angles(peds[:self.num_pedestrians // 2], GOALS_RIGHT),
                    get_angles(peds[self.num_pedestrians // 2:], GOALS_LEFT)
                    ])
        weighted_angles = np.sum(expected_angles*belief, axis=1)
        weighted_directions = get_pedestrian_directions(speeds, weighted_angles)
        obs[-self.num_pedestrians*2:] = (peds + weighted_directions).ravel()
        obs = np.concatenate([obs, belief.ravel()])

        return obs


    def reset(self):
        obs = self.env.reset()
        bel = self.estimator.estimate(None, obs)
        obs = self._expand_belief(obs, bel)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Estimate
        belief = self.estimator.estimate(action, obs, **info)
        obs = self._expand_belief(obs, belief, **info)
        return obs, reward, done, info


if __name__ == "__main__":
    env = BayesCrossWalkEnv()
    obs = env.reset()
    for _ in range(200):
        obs, _, _, _ = env.step(env.action_space.sample())
        print(np.around(obs,1))

    import IPython; IPython.embed()

