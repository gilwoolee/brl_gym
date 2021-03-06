import math
import numpy as np

import gym
from gym.spaces import Box
from gym import utils
from gym.utils import seeding

from matplotlib import pyplot as plt

# Implementation of Rob Platt's https://dspace.mit.edu/handle/1721.1/62571
# Room is [-1, -2] x [7, 4] (8 by 6)
# Zero process noise, x in R^2, u in R^2
# f(x, u) = x + u
# Observatoin is identity + noise
# g(x) = x + w
# with w ~ N(0, w(x)) being zero-mean Gaussian observation noise,
# w(x) = 0.5(5 - x1)**2 + const.
# Belief b = (x1, x2, s) where s is covariance scalar.
# Total cost is set as sLs + \sum_{traj} (x'Qx + u'Ru) but
# since env is not aware of s, it only gives step rewards, -(x'Qx + u'Ru).
# Goal pose is randomly sampled from the grid.
# Init_ropot_pose is sampled from N([2.5,0], 2)
class LightDark(gym.Env, utils.EzPickle):

    def __init__(self, init_robot_pose=None, goal_pose=None):
        self.action_min = np.array([-1,-1]) * 0.5
        self.action_max = np.array([1, 1]) * 0.5
        self.pos_min = np.array([-1, -2])
        self.pos_max = np.array([7, 4])
        self.goal_min = np.array([0, -2])
        self.goal_max = np.array([2, 4])
        self.dist_min = np.array([-8, -6])
        self.dist_max = np.array([8, 6])
        self.init_min = np.array([2, -2])
        self.init_max = np.array([4, 4])

        # cost terms
        self.R = 0.5
        self.Q = 0.5
        self.QT = 5000

        self.init_x = None
        self.init_goal = None

        self.action_space = Box(self.action_min, self.action_max,
            dtype=np.float32)
        self.observation_space = Box(
                np.concatenate([self.pos_min, self.goal_min, self.dist_min, [0], [0]]),
                np.concatenate([self.pos_max, self.goal_max, self.dist_max, [8], [self._get_noise_std([-1, 0])]]),
                dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Randomize has no effect here
    def reset(self, randomize=False):
        if self.init_x is not None:
            self.x = self.init_x
            self.goal = self.init_goal
        else:
            init_box = self.init_max - self.init_min
            init_center = ( self.init_max + self.init_min ) / 2.0
            self.x = self.np_random.uniform(low=-0.5, high=0.5, size=2)* init_box + init_center
            self.x = np.clip(self.x, self.init_min, self.init_max)
            goal_box = self.goal_max - self.goal_min
            goal_center = ( self.goal_max + self.goal_min ) / 2.0
            self.goal = self.np_random.uniform(low=-0.5, high=0.5, size=2) * goal_box + goal_center
            self.goal = np.clip(self.goal, self.goal_min, self.goal_max)
        self.timestep = 0
        return self._get_obs(self.x)

    def set_start_and_goal(self, start, goal):
        self.init_x = start
        self.init_goal = goal
        self.timestep = 0
        return self.reset()

    def _get_noise_std(self, x):
        noise_std = np.sqrt(((5 - x[0])**2) / 4.0) + 1e-6 # Originally division by 2.0
        return noise_std

    def _get_obs(self, x):
        noise_std = self._get_noise_std(x)
        assert noise_std > 0, x
        noise = self.np_random.normal(0, noise_std, 2)
        obs = np.clip(x + noise, self.pos_min, self.pos_max)
        return np.concatenate([obs, self.goal, self.goal - obs, [5.0 - x[0]], [noise_std]])

    def step(self, action, update=True):
        action = np.clip(action * 0.5, self.action_min, self.action_max)
        x = self.x + action
        x = np.clip(x, self.pos_min, self.pos_max)
        cost = np.sum((x - self.goal)**2) * self.Q + np.sum(action**2) * self.R

        obs = self._get_obs(x)

        if update:
            self.timestep += 1
            self.x = x

        dist_to_goal = np.linalg.norm(x - self.goal, ord=2)
        if dist_to_goal < 1e-1:
            done = True
            cost = np.sum((x - self.goal)**2) * self.QT
        else:
            done = False

        return obs, -cost, done, {}


    def render(self, mode='human'):
        print ("Last Obs: ", "LEFT" if self.last_obs == 0 else "RIGHT", \
            "\tHidden Tiger: ", "LEFT" if self.tiger == 0 else "RIGHT")
        pass

    def visualize(self, pos_history, belief_history=None,
            filename=None, show=False):
        pos_history = np.array(pos_history)
        fig = plt.figure()

        plt.imshow([[192,64], [192,64]],
            cmap = plt.cm.Greys,
            interpolation = 'gaussian',
            extent=(-1, 5, -2, 4),
            vmin=0, vmax=255
            )
        plt.imshow([[64,192], [64,192]],
            cmap = plt.cm.Greys,
            interpolation = 'gaussian',
            extent=(5, 7, -2, 4),
            vmin=0, vmax=255
            )

        # Draw boundaries
        plt.plot([-1, 7], [-2, -2], linewidth=5, color='k')
        plt.plot([-1, 7], [4, 4],linewidth=5, color='k')
        plt.plot([-1, -1], [-2, 4], linewidth=5, color='k')
        plt.plot([7, 7], [-2, 4], linewidth=5, color='k')

        # Draw trajectory
        pos_history = np.asarray(pos_history)
        plt.plot(pos_history[:, 0], pos_history[:, 1],
                color='g', linewidth=3, label='Path')

        # Draw start and goal
        plt.plot(pos_history[0, 0], pos_history[0, 1], marker='o',
                color='r', label='start')
        plt.plot(pos_history[-1, 0], pos_history[-1, 1], marker='*',
                color='r', label='end')

        # Draw belief
        if belief_history is not None:
            for belief in belief_history:
                circle = plt.Circle(belief[:2], np.sqrt(belief[-1]), color='b', alpha=0.1)
                plt.gca().add_artist(circle)

        plt.plot(self.goal[0], self.goal[1], marker='o', color='y', label='goal')

        plt.axis('equal')
        plt.xlim((-1, 7))
        plt.ylim((-2, 4))
        plt.xticks(np.arange(-1, 8))
        plt.yticks(np.arange(-2, 5))

        plt.grid()
        plt.legend(loc='upper right')

        if filename is not None:
            plt.savefig(filename)
            print("Saved at {}".format(filename))

        if show:
            plt.show()


if __name__ == "__main__":

    env = LightDark()

    pos_history = []
    for i in range(5):
        pos_history += [env.x.copy()]
        print (env.step(np.array([1.0,0.5])))

    for i in range(5):
        pos_history += [env.x.copy()]
        print(env.step(np.array([-0.5,0.5])))

    env.visualize(pos_history, show=True)
