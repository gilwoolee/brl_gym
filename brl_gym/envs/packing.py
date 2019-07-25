import numpy as np

import gym
from gym.spaces import MultiDiscrete
from gym.utils import seeding
from gym import utils

from matplotlib import pyplot as plt
import mpld3


GRID_SIZE = 10
OBJ_COUNT = 3


class Packing(gym.Env, utils.EzPickle):
    def __init__(self):
        self.seed()
        # No rotation yet
        self.action_space = MultiDiscrete([OBJ_COUNT, GRID_SIZE, GRID_SIZE])

        # State space: TODO
        self.observation_space = None

    def render(self, mode='human'):
        if mode == 'human':
            print(self.box)
        elif mode == 'web':
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0,0].imshow(self.box, interpolation='none')
            ax[0,1]
            plt.show()
            # mpld3.show() showing in server


    def reset(self):
        self.box = np.ones((GRID_SIZE, GRID_SIZE)) * -1

        self.packed_items = dict()
        self._load_objects()
        return self._get_obs()

    def _load_objects(self):
        self.unpacked_items = dict()
        # Obj id starts from 0
        self.unpacked_items[0] = (3,3)
        self.unpacked_items[1] = (2,2)
        self.unpacked_items[2] = (4,1)

    def _place(self, obj_id, x, y, dry_run=False):
        if obj_id not in self.unpacked_items:
            return False

        sx, sy = self.unpacked_items[obj_id]

        # check bounds
        if x < 0 or x + sx >= GRID_SIZE or y < 0 or y + sy >= GRID_SIZE:
            return False

        # check collison
        if not self._collision(obj_id, x, y):
            # place
            if not dry_run:
                self.box[x: x + sx, y: y + sy] = obj_id
                self.unpacked_items.pop(obj_id)
                self.packed_items[obj_id] = (sx, sy)
            return True

    def _collision(self, obj_id, x, y):
        assert obj_id in self.unpacked_items

        sx, sy = self.unpacked_items[obj_id]
        return np.any(self.box[x: x + sx, y: y + sy] != -1)

    def step(self, action):
        assert len(action) == 3
        print(action)

        # bounds
        if (action[0] >= OBJ_COUNT or
                action[1] >= GRID_SIZE or action[1] < 0 or
                action[2] >= GRID_SIZE or action[2] < 0):
            return self._get_obs(), 0, False, {}

        result = self._place(action[0], action[1], action[2], False)
        reward = 1 if result else 0

        done = len(self.unpacked_items) == 0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.box, self.unpacked_items

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    env = Packing()
    o = env.reset()
    o, r, d, info = env.step(env.action_space.sample())
    env.render(mode='web')
    import IPython; IPython.embed()
