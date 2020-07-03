import gym


from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from brl_gym.envs.util import fig2np

# Crosswalk is 4 x 4 space, [0, 0] x [4, 4]
# Driver is approaching the crosswalk from (2, -1)
# Driver actions are (velocity, steering angle).
# 6 pedestrians, 3 on left / 3 on right. 3 goals on each side.
# Pedestrians start anywhere vertically, y in [1, 6]. x is either 0 or 4.
# And go straight to their goals, with fixed velocity,
# uniformly sampled from [0.2 m/s, 0.8  m/s].
# Done when the car reaches y = 4.5.
# Angles are ccw, with 0 rad correponding to y-axis, the initial direction of the car.

#colors = ['#C7980A', '#F4651F', '#82D8A7', '#CC3A05', '#575E76', '#156943']
#colors = ['#88CCEE','#DDCC77','#117733','#332288','#44AA99','#661100']
#colors = ['#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628', '#e41a1c']
from matplotlib import cm
tab10 = cm.get_cmap('tab10')
colors = np.vstack([tab10.colors[:3], tab10.colors[4:6], tab10.colors[7], tab10.colors[3]])

# Crosswalk but with (velocity, steering angle) control
class CrossWalkVelEnv(gym.Env):
    def __init__(self, timestep=0.1):
        self.action_space = spaces.Box(np.array([0.0, -0.5]), np.array([0.6, 0.5]))
        self.car_length = 0.33 # Length of the MuSHR car
        self.num_pedestrians = 4
        self.timestep = timestep
        self.observation_space = spaces.Box(low=np.ones(24)*-10.0,
                                            high=np.ones(24)*10.0,
                                            dtype=np.float32)
        self.x_limit = np.array([0, 4], dtype=np.float32)
        self.y_limit = np.array([0, 4], dtype=np.float32)
        self.car_y_goal = 4.5
        self.x_left = 0.0
        self.x_right = 4.0
        self.y_starts = np.arange(1.0, 5.0, 1.0) # discretized to avoid overlap
        self.ped_speed_limits = np.array([0.2, 0.4], dtype=np.float32)
        self.car_start_y = 0.0
        self.car_speed_limits = np.array([0.2, 0.4], dtype=np.float32)
        self.car_steering_limits = np.array([-0.5,0.5], dtype=np.float32)
        self.goal_xs = np.concatenate([np.ones(2)*self.x_right, np.ones(2)*self.x_left]).ravel()


    def reset(self):
        # Hidden intention of pedestrians
        self.goals = np.vstack([self.goal_xs, [np.random.choice(3, size=self.num_pedestrians) + 0.5]]).transpose()
        # Initial position of pedestrians (Each row is a pedestrian)
        ped_left = np.vstack([self.x_left * np.ones(2), np.random.choice(self.y_starts, size=2, replace=False)]).transpose()
        ped_right = np.vstack([self.x_right * np.ones(2), np.random.choice(self.y_starts, size=2, replace=False)]).transpose()

        self.pedestrians = np.vstack([ped_left, ped_right])

        # Pedestrians have fixed speed, but can change directions
        self.pedestrian_speeds = np.random.uniform(
            self.ped_speed_limits[0], self.ped_speed_limits[1], size=self.num_pedestrians)
        self.pedestrian_angles = self._get_pedestrian_angles()

        # Agent's initial position, speed, angle
        self.pose = np.array([np.random.uniform(1, 3), 0.0, np.random.uniform(-0.5, 0.5)])
        self.speed = np.random.uniform(self.car_speed_limits[0], self.car_speed_limits[1])
        self.steering_angle = np.random.uniform(self.car_steering_limits[0], self.car_steering_limits[1])
        self.car_front = self.pose[:2] + \
                         self.car_length * np.array([-np.sin(self.pose[2]), np.cos(self.pose[2])])
        self.pedestrian_directions = self._get_pedestrian_directions(self.pedestrian_angles)

        self.fig = None
        self.car = None
        # import IPython; IPython.embed(); import sys; sys.exit(0)
        return self.get_obs()

    def _get_pedestrian_angles(self, add_noise=True):
        # Angles are directed straight to the goals
        diff = self.goals - self.pedestrians
        angles = -1.0*np.arctan2(diff[:,0], diff[:,1])
        if add_noise:
            angles += np.random.normal(size=self.pedestrians.shape[0], scale=0.5)
        return angles

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.speed = action[0]
        self.pose[2] += action[1] * self.timestep
        done = False

        # Penalize time & distance to goal. y = 4.0
        reward = -0.5 - np.abs(self.car_y_goal - self.pose[1]) * 0.1

        # move car
        self.pose[:2] += self.timestep*self.speed*np.array([-np.sin(self.pose[2]), np.cos(self.pose[2])])
        self.car_front = self.pose[:2] + \
                         self.car_length * np.array([-np.sin(self.pose[2]), np.cos(self.pose[2])])

        # angle penalty
        reward -= np.abs(self.pose[2]) * 0.1

        # move pedestrians
        for i in range(self.num_pedestrians//2):
            if self.pedestrians[i,0] >= self.x_limit[1]:
                self.pedestrian_speeds[i] = 0.0
        for i in range(self.num_pedestrians//2, self.num_pedestrians):
            if self.pedestrians[i,0] <= self.x_limit[0]:
                self.pedestrian_speeds[i] = 0.0

        self.pedestrian_angles = self._get_pedestrian_angles()
        self.pedestrian_directions = self._get_pedestrian_directions(self.pedestrian_angles)
        self.pedestrians += self.pedestrian_directions * self.timestep

        # Collision
        if (np.any(np.linalg.norm(self.car_front - self.pedestrians, axis=1) < 0.3) or
                np.any(np.linalg.norm(self.pose[:2] - self.pedestrians, axis=1) < 0.3)):
            done = True
            # print("bad")
            reward -= 10*(1.0+self.speed)**2

        if self.car_front[0] <= self.x_limit[0] + 0.5\
                or self.car_front[0] >= self.x_limit[1] - 0.5\
                or self.car_front[1] <= -0.5:
            reward += -5.0
            # print("bad")
            done = True

        elif self.pose[1] >= self.car_y_goal:
            reward += 100.0
            # print("good")
            done = True
        return self.get_obs(), reward, done, dict(
                pedestrians=self.pedestrians,
                pedestrian_speeds = self.pedestrian_speeds,
                pedestrian_angles = self.pedestrian_angles)

    def _get_pedestrian_directions(self, angles):
        speeds = self.pedestrian_speeds.reshape(-1,1)
        speeds += np.clip(np.random.normal(size=speeds.shape)*0.1, 0.0, 0.1)
        return speeds \
                * np.array([-np.sin(angles),
                            np.cos(angles)]).transpose()

    def get_obs(self):
        car_direction = self.car_front - self.pose[:2]

        obs = np.concatenate([
            self.pose[:2].ravel(),
            self.car_front.ravel(),
            (self.car_front + car_direction).ravel(),
            [self.speed, self.pose[2]],
            self.pedestrians.ravel(),
            (self.pedestrians + self.pedestrian_directions).ravel()])
        return obs

    def _visualize(self, show=False, filename=None, nparray=False, head_only=False, transparent=True):
        # if filename is None:
        #     plt.ion()
        #     self.fig = plt.figure()
        fig = plt.figure()

        # Draw goal
        plt.plot([0, 4], [self.car_y_goal, self.car_y_goal], linewidth=3, color='r')

        # Draw boundaries
        plt.plot([0, 0], [-1, 5], linewidth=2, color='k', linestyle='--')
        plt.plot([4, 4], [-1, 5], linewidth=2, color='k', linestyle='--')

        plt.axis('square')
        plt.axis('off')
        plt.xlim((-0.1, 4.1))
        plt.ylim((-0.5, self.car_y_goal+0.1))

        # Car
        # Pose of the end of the car
        car = self.pose.copy()
        angle = car[2]
        if self.car is not None:
            self.car.remove()

        if not head_only:
            self.car = Rectangle((car[0], car[1]),
                        0.1, self.car_length, angle=np.rad2deg(angle), color=colors[-1])
        else:
            self.car = Circle([car[0], car[1]], radius=0.1, color=colors[-1], zorder=25)

        plt.gca().add_patch(self.car)

        # pedestrians
        for i in range(len(self.pedestrians)):
            ped = Circle(self.pedestrians[i], radius=0.2, color=colors[i], zorder=20)
            plt.gca().add_patch(ped)

            xy = self.pedestrians[i]
            dxdy = self.pedestrian_directions[i]

            #rect = Rectangle((xy[0], xy[1]),
            #        0.1, 1.0, angle=np.rad2deg(self.pedestrian_angles[i]), color=colors[i])
            #plt.gca().add_patch(rect)
        """
        for i in range(2):
            goal = Circle(self.goals[i], radius=0.25, color='g', zorder=10)
            plt.gca().add_patch(goal)
        """

        if nparray:
            raise NotImplementedError
            return fig2np(fig)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', transparent=transparent)

        if show:
            fig.canvas.draw()

    def _visualize_history(self, states, actions):
        pass

    def render(self, mode='human'):
        self._visualize(show=True)

if __name__ == "__main__":
    env = CrossWalkVelEnv()

    states = []
    actions = []


    obs = env.reset()
    states.append(dict(x=env.x.copy(), car_front=env.car_front.copy(), ped=env.pedestrians.copy()))

    print("obs", obs)
    done = False

    env._visualize(show=False, filename="imgs/crosswalk.png", head_only=True, transparent=False)
    for t in range(600):
        obs, rew, done, info = env.step(env.action_space.sample())
        env._visualize(show=False, filename="imgs/crosswalk_{}.png".format(t), head_only=True)
        if done:
            break


        print("obs", np.around(obs, 1))
        print("rew", rew)
        print("done", done)

