import math
import numpy as np

import gym
from gym.spaces import Box
from gym import utils
from gym.utils import seeding

from matplotlib import pyplot as plt

# Harder version of the regular LightDark.
# There is no observation unless the agent is within 0.5 unit from x=5
#
# Implementation of Rob Platt's https://dspace.mit.edu/handle/1721.1/62571
# Room is [-1, -8] x [7, 10] (8 by 18)
# Zero process noise, x in R^2, u in R^2
# f(x, u) = x + u
# Observation is (0,0) if |x - 5| > 0.5
# g(x) = (0,0)
# If |x - 5| <= 0.5
# g(x) = x + w
# with w ~ N(0, w(x)) being zero-mean Gaussian observation noise,
# w(x) = 0.5(5 - x1)**2 + const.
# Belief b = (x1, x2, s) where s is covariance scalar.
# Total cost is set as sLs + \sum_{traj} (x'Qx + u'Ru) but
# since env is not aware of s, it only gives step rewards, -(x'Qx + u'Ru).
# Goal pose is randomly sampled from the grid.
# Init_ropot_pose is sampled from N([2.5,0], 2)

# Goal is either [-0.5, 9.5] or [-0.5, -7.5]
# GOALS = np.array([[-9.5, 0.5], [-9.5, -0.5]])
light = np.array([0.0, 9.0])

class LightDarkHard(gym.Env, utils.EzPickle):

    def __init__(self, init_robot_pose=None, goal_pose=None):
        self.action_min = np.array([-1,-1])
        self.action_max = np.array([1, 1])
        self.pos_min = np.array([-10, -10])
        self.pos_max = np.array([10, 10])
        self.init_min = np.array([-2, -2])
        self.init_max = np.array([2, 2])
        self.max_dist_to_light = 20.0
        # cost terms
        self.R = 0.5
        self.Q = 0.5

        self.action_space = Box(self.action_min, self.action_max,
            dtype=np.float32)
        self.observation_space = Box(
                np.concatenate([self.pos_min, np.array([-10,-10]), [-1], [-1]]),
                np.concatenate([self.pos_max, np.array([-8, 10]),  [20], [20.0]]),
                dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Randomize has no effect here
    def reset(self, randomize=False):
        self.goal = self.np_random.uniform(low=-0.5, high=0.5, size=2) \
                    * np.array([2.0, 10.0]) + np.array([-9.0, 0.0])
        init_box = self.init_max - self.init_min
        init_center = ( self.init_max + self.init_min ) / 2.0
        self.x = self.np_random.uniform(low=-0.5, high=0.5, size=2)* init_box + init_center
        self.x = np.clip(self.x, self.init_min, self.init_max)
        self.timestep = 0
        return self._get_obs(self.x)

    def set_start_and_goal(self, start, goal):
        self.init_x = start
        self.init_goal = goal
        self.timestep = 0
        return self.reset()

    def _get_noise_std(self, x):
        dist_to_light = self._get_dist_to_light(x)

        noise_std = 0.5*dist_to_light + 1e-6 # Originally division by 2.0
        #noise_std = 0.01
        if np.abs(dist_to_light) <= self.max_dist_to_light:
            return noise_std
        else:
            return self.max_dist_to_light

    def _get_dist_to_light(self, x):
        return np.abs(light[1] - x[1])

    def _get_obs(self, x):
        dist_to_light = self._get_dist_to_light(x)
        if np.abs(dist_to_light) <= self.max_dist_to_light:
            noise_std = self._get_noise_std(x)
            assert noise_std > 0, x
            noise = self.np_random.normal(0, noise_std, 2)
            obs = np.clip(x + noise, self.pos_min, self.pos_max)
            return np.concatenate([obs, self.goal, [dist_to_light], [noise_std]])
        else:
            return np.concatenate([np.array([0,0]), self.goal, [dist_to_light], [20.0]])

    def step(self, action, update=True):
        action = np.clip(action, self.action_min, self.action_max)
        x = self.x + action * 2.0
        x = np.clip(x, self.pos_min, self.pos_max)
        cost = 0.5*np.sum((x - self.goal)**2) * self.Q + 0.5*np.sum(action**2) * self.R
        cost *= 0.5

        obs = self._get_obs(x)

        self.timestep += 1
        self.x = x

        dist_to_goal = np.linalg.norm(x - self.goal, ord=2)

        if dist_to_goal < 0.5:
            done = True
            cost = -100.0
        else:
            done = False

        return obs, -cost, done, dict(noise=self._get_noise_std(x))


    def render(self, mode='human'):
        # print ("Last Obs: ", "LEFT" if self.last_obs == 0 else "RIGHT", \
            # "\tHidden Tiger: ", "LEFT" if self.tiger == 0 else "RIGHT")
        self.visualize([self.x], show=True)
        pass

    def visualize(self, pos_history, belief_history=None,
            filename=None, show=False):
        pos_history = np.array(pos_history)
        fig = plt.figure(figsize=(8,8))
        """
        plt.imshow([[192,64], [192,64]],
            cmap = plt.cm.Greys,
            interpolation = 'gaussian',
            extent=(-1, 5, -8, 10),
            vmin=0, vmax=255
            )
        plt.imshow([[64,192], [64,192]],
            cmap = plt.cm.Greys,
            interpolation = 'gaussian',
            extent=(5, 7, -8, 10),
            vmin=0, vmax=255
            )
        """
        print("pos history", pos_history)

        # Draw boundaries
        plt.plot([-10, 10], [-10, -10], linewidth=5, color='k')
        plt.plot([-10, 10], [10, 10],linewidth=5, color='k')
        plt.plot([-10, -10], [-10, 10], linewidth=5, color='k')
        plt.plot([10, 10], [-10, 10], linewidth=5, color='k')

        # Draw dark region
        rectangle = plt.Rectangle((-10.0,-10.0), width=(19.0 - self.max_dist_to_light), height=20.0, fc='k', alpha=0.1)
        #rectangle = plt.Rectangle((8.,-10.0), width=2.0, height=20.0, fc='r', alpha=0.3)
        plt.gca().add_patch(rectangle)

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
                plt.add_artist(circle)

        plt.plot(self.goal[0], self.goal[1], marker='o', color='g', label='goal')
        plt.plot(light[0], light[1], marker='o', color='y', label='light')

        plt.axis('equal')
        plt.xlim((-10, 10))
        plt.ylim((-10, 10))
        plt.xticks(np.arange(-10, 11))
        plt.yticks(np.arange(-10, 11))

        plt.grid()
        plt.legend(loc='upper right')

        if filename is not None:
            plt.savefig(filename)
            print("Saved at {}".format(filename))

        if show:
            plt.show()


if __name__ == "__main__":

    env = LightDarkHard()

    pos_history = []
    for i in range(5):
        pos_history += [env.x.copy()]
        print (env.step([1.0,0.5]))

    for i in range(5):
        pos_history += [env.x.copy()]
        print(env.step([-0.5,0.5]))

    env.visualize(pos_history, show=True)
