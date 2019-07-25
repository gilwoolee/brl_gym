import numpy as np

from brl_gym.estimators.estimator import Estimator
from brl_gym.envs.herb_table import HerbTable, Action, GridCellState, ACTION_NAME


class BayesHerbTableEstimator(Estimator):
    """
    This class estimates herb_table given a history of observations
    It assumes that the beliefs and targets occupy only single grid
    """
    def __init__(self):
        env = HerbTable()
        # probability grids
        self.grid_shape = env.grid.shape
        super(BayesHerbTableEstimator, self).__init__(env.observation_space, env.action_space)

        # First layer is for beliefs
        # Second is for targets
        self.belief_high = np.ones((2, env.grid.shape[0], env.grid.shape[1]), dtype=np.float32)
        self.belief_low = np.zeros((2, env.grid.shape[0], env.grid.shape[1]), dtype=np.float32)
        self.nominal_env = env

    def reset(self):
        # Each column sums to 1
        self.belief = np.ones(self.grid_shape, dtype=np.float32) / self.grid_shape[0]
        return self.get_belief()

    def estimate(self, action, observation, **kwargs):
        """
        Given the latest (action, observation), update belief
        """
        if action is not None:
            action_type, x, y = self.nominal_env._decode_action(action)
            if action_type == Action.MOVE_LEFT:
                if kwargs['action_success']:
                    self.belief[:-1, y] = self.belief[1:, y]
                    self.belief[-1, y] = 0.0
                elif x > 0:
                    # Failure due to the cell being empty
                    self.belief[x, y] = 0.0

            if action_type == Action.MOVE_RIGHT:
                if kwargs['action_success']:
                    self.belief[1:, y] = self.belief[:-1, y]
                    self.belief[0, y] = 0.0
                elif x < self.grid_shape[0] - 1:
                    # Failure due to the cell being empty
                    self.belief[x, y] = 0.0

            if action_type == Action.GRASP:
                if kwargs['action_success']:
                    self.belief[:, y] = 0.0
                else:
                    # Failure due to the cell being empty
                    self.belief[x, y] = 0.0

            if np.sum(self.belief[:, y]) > 0:
                self.belief[:, y] /= np.sum(self.belief[:, y])

        # Any concrete observations are used to update the grids
        for i in range(self.belief.shape[1]):
            if np.any(observation[:, i] == GridCellState.MOVABLE):
                self.belief[:, i] =  0.0
                self.belief[observation[:, i] == GridCellState.MOVABLE, i] =  1.0

            if np.any(observation[:, i] == GridCellState.TARGET):
                self.belief[:, i] =  0.0
                self.belief[observation[:, i] == GridCellState.TARGET, i] =  1.0

        for i in range(self.belief.shape[1]):
            if np.any(self.belief[:, i] == 1.0):
                continue
            self.belief[observation[:, i] == GridCellState.EMPTY, i] = 0.0
            if np.sum(self.belief[:, i]) > 0:
                self.belief[:, i] /= np.sum(self.belief[:, i])

        return self.get_belief()

    def get_belief(self):
        movable = self.belief.copy()
        movable[:, 3:] = 0.0
        target = self.belief.copy()
        target[:, :3] = 0.0
        return np.array([movable, target])


if __name__ == "__main__":
    env = HerbTable()
    estimator = BayesHerbTableEstimator()
    obs = env.reset()
    belief = estimator.reset()
    print(obs)
    print(belief)
    belief = estimator.estimate(None, obs)


    import IPython; IPython.embed()
