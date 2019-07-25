from brl_gym.wrapper_envs.wrapper_env import WrapperEnv

from brl_gym.envs.herb_table import HerbTable, colors, GridCellState
from brl_gym.estimators.bayes_herbtable_estimator import BayesHerbTableEstimator

import gym
from gym import utils
from gym.spaces import Box
import numpy as np


class BayesHerbTable(WrapperEnv):
    def __init__(self):
        env = HerbTable()
        estimator = BayesHerbTableEstimator()
        super(BayesHerbTable, self).__init__(env, estimator)
        self.observation_space = Box(self.estimator.belief_low,
                                     self.estimator.belief_high, dtype=np.float32)
        self.viewer = None

    def _augment_observation(self,
                             action,
                             obs,
                             **kwargs):
        # Estimate
        self.estimator.estimate(
                action, obs, **kwargs)
        return self.estimator.get_belief(), kwargs


    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        grid = self.estimator.get_belief()
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000,500)
            self.viewer.set_bounds(0, 10, 0, 5)

        for i in range(10):
            self.viewer.draw_line((i,0), (i, 5))
        for i in range(5):
            self.viewer.draw_line((0, i), (10, i))

        # self.viewer.draw_polygon([(0, 0), (0, 5),(10,5),(10,0)],
                                 # color=[1, 0, 0])

        for i in range(grid.shape[1]):
            for j in range(grid.shape[2]):

                if grid[0, i, j] > 0:
                    self.viewer.draw_polygon([(i,j), (i,j+1),(i+1,j+1),(i+1,j)],
                        color=np.array(colors[int(GridCellState.MOVABLE)])*grid[0, i, j])
                elif grid[1, i, j] > 0:
                    self.viewer.draw_polygon([(i,j), (i,j+1),(i+1,j+1),(i+1,j)],
                        color=np.array(colors[int(GridCellState.TARGET)])*grid[1, i, j])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

if __name__ == "__main__":
    print("=============== Bayes ===============")
    env = BayesHerbTable()
    obs = env.reset()
    env.render()
    import IPython; IPython.embed()

