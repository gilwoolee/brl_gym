import numpy as np

import gym
from gym.spaces import Discrete
from gym import utils
from gym.utils import seeding

class GridCellState:
    EMPTY = 0
    TARGET = 1
    MOVABLE = 2
    INVISIBLE = 3

class Action:
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    GRASP = 2

ACTION_NAME = {
    Action.MOVE_LEFT: "MOVE_LEFT",
    Action.MOVE_RIGHT: "MOVE_RIGHT",
    Action.GRASP: "GRASP"
    }

colors = {GridCellState.EMPTY: [1, 1, 1],
          GridCellState.TARGET:[0, 1, 0],
          GridCellState.MOVABLE:[0, 0, 1],
          GridCellState.INVISIBLE:[.5, .5, .5]}

class HerbTable(gym.Env, utils.EzPickle):
    def __init__(self):
        self.grid = np.zeros((10, 5), dtype=np.int)
        self.nA = 3 * self.grid.size
        self.nS = self.grid.size * 4
        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)

        self.block_sizes = [1, 1, 1, 1, 1]
        self.block_starts = []
        self.seed()
        self.viewer = None

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        grid = self._get_obs()
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000,500)
            self.viewer.set_bounds(0, 10, 0, 5)

        for i in range(10):
            self.viewer.draw_line((i,0), (i, 5))
        for i in range(5):
            self.viewer.draw_line((0, i), (10, i))

        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                self.viewer.draw_polygon([(i,j), (i,j+1),(i+1,j+1),(i+1,j)],
                    color=colors[int(grid[i,j])])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.grid = np.ones((10, 5), dtype=np.int) * GridCellState.EMPTY

        self.block_starts = []
        # Place the blocks in the first three rows and then targets
        for i in range(5):
            start = self.np_random.choice(self.grid.shape[0] - self.block_sizes[i] + 1)
            value = GridCellState.MOVABLE if i < 3 else GridCellState.TARGET
            self.grid[start : start + self.block_sizes[i], i] = value
            self.block_starts += [start]

        return self._get_obs()

    def step(self, action):
        reward = -2
        done = False

        action_type, x, y = self._decode_action(action)
        action_success = False
        if action_type == Action.GRASP:
            if self.grid[x, y] == GridCellState.TARGET:
                reward = 10
                action_success = True
                self.grid[x, y] = GridCellState.EMPTY
            else:
                reward = -10
        else:
            if self.grid[x, y] == GridCellState.EMPTY:
                pass
            elif action_type == Action.MOVE_LEFT and self.grid[0, y] != GridCellState.EMPTY:
                pass
            elif x == self.grid.shape[0] and action_type == Action.MOVE_RIGHT:
                pass
            else:
                move = -1 if action_type == Action.MOVE_LEFT else 1
                self.grid[:, y] = GridCellState.EMPTY
                value = GridCellState.MOVABLE if y < 3 else GridCellState.TARGET
                start = np.clip(self.block_starts[y] + move,
                                0,
                                self.grid.shape[0] - self.block_sizes[y])
                end = start + self.block_sizes[y]
                self.grid[start : end, y] = value
                action_success = self.block_starts[y] != start
                self.block_starts[y] = start
                reward = 0

        if not np.any(self.grid == GridCellState.TARGET):
            done = True

        return self._get_obs(), reward, done, {"action_success": action_success}

    def _get_obs(self):
        visible_grid = self.grid.copy()

        for y in range(1, self.grid.shape[1]):
            blocked = visible_grid[:, y - 1] != GridCellState.EMPTY
            visible_grid[blocked, y] = GridCellState.INVISIBLE

        return visible_grid

    def _decode_action(self, action):
        action_type = action // self.grid.size
        xy = action - action_type * self.grid.size
        x = xy // self.grid.shape[1]
        y = xy - x * self.grid.shape[1]
        return action_type, x, y

    def _encode_action(self, action_type, x, y):
        xy = x * self.grid.shape[1] + y
        action = action_type * self.grid.size + xy
        return action


if __name__ == "__main__":
    env = HerbTable()
    o1 = env.reset()
    print(env.grid)
    env.render(mode='rgb_array')
    action = env._encode_action(Action.MOVE_RIGHT, 6,4)
    print(env._decode_action(action))
    o, r, d, i = env.step(action)
    import IPython; IPython.embed()

