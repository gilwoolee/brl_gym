import gym


from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from brl_gym.envs.util import fig2np

# Crosswalk is 10 x 5 space, [0, 0] x [9, 4]
# Driver is approaching the crosswalk from (4, -5)
# Driver actions are (accel, delta-angle).
# 2 pedestrians, 4 goals, 2 on each side
# Goals [0, 0], [0, 4], [9, 0], [9, 4]
# Pedestrians start anywhere on [0, 0] x [0, 4] and [9, 0] x [9, 4]
# And go straight to their goals, with fixed velocity.
# Collision results in a large penalty
# Rcol = -1000 * (v^2 + 0.5)
# Rtime = -0.1
# Done when the car crosses the crosswalk
# Angles are ccw, with 0 rad correponding to y-axis, the initial direction of the car.

GOAL_LEFT  = np.array([[0, 0], [0, 4]])
GOAL_RIGHT = np.array([[9, 0], [9, 4]])

#colors = ['#C7980A', '#F4651F', '#82D8A7', '#CC3A05', '#575E76', '#156943']

#colors = ['#88CCEE','#DDCC77','#117733','#332288','#44AA99','#661100']
#colors = ['#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628', '#e41a1c']
from matplotlib import cm
tab10 = cm.get_cmap('tab10')
colors = np.vstack([tab10.colors[:3], tab10.colors[4:6], tab10.colors[7], tab10.colors[3]])

class CrossWalkEnv(gym.Env):
    def __init__(self, use_vision=False):
        self.action_space = spaces.Box(np.array([-1.2, -1.2]), np.array([1, 1]))
        self.car_length = 2.0
        self.use_vision = use_vision
        self.num_pedestrians = 6

        if not use_vision:
            self.observation_space = spaces.Box(-np.ones(32)*-10.0,
                                                high=np.ones(32)*10.0,
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Dict(dict(obs=[], img=[]))
        self.fig = None

    def reset(self):
        # Hidden intention of pedestrians
        self.goals = np.vstack([[9.0, 9, 9, 0, 0, 0], [np.random.choice(3, size=self.num_pedestrians) + 0.5]]).transpose()

        # Initial position of pedestrians (Each row is a pedestrian)
        self.pedestrians = np.vstack([[-0.5, -0.5, -0.5, 9.5, 9.5, 9.5],
                                        np.random.uniform(size=self.num_pedestrians)*4.0]).transpose()
        # Pedestrians have fixed speed, but can change directions
        self.pedestrian_speeds = np.clip(np.random.uniform(size=self.pedestrians.shape[0]), 0.1, 1.0)
        self.pedestrian_angles = self._get_pedestrian_angles()


        # Agent's initial position, speed, angle
        self.x = np.array([4.0,-5.0])
        self.speed = np.clip(np.random.uniform(size=1)*0.4, 0.0, 0.4)[0]
        self.angle = 0.0
        self.car_front = self.x + \
                         self.car_length * np.array([-np.sin(self.angle), np.cos(self.angle)])
        self.pedestrian_directions = self._get_pedestrian_directions(self.pedestrian_angles)


        # plt.ion()
        # self.fig = plt.figure()

        self.fig = None
        self.car = None
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
        action *= 0.1
        self.speed += action[0]
        self.angle += action[1]

        done = False
        # Time & distance to goal. y = 4.0
        reward = -0.1 - (4.0 - self.x[1])

        # move car
        self.x += self.speed*np.array([-np.sin(self.angle), np.cos(self.angle)])
        self.car_front = self.x + \
                         self.car_length * np.array([-np.sin(self.angle), np.cos(self.angle)])

        # move pedestrians
        for i in range(3):
            if self.pedestrians[i,0] >= 9.0:
                self.pedestrian_speeds[i] = 0.0
        for i in range(3, 6):
            if self.pedestrians[i,0] <= 0.0:
                self.pedestrian_speeds[i] = 0.0

        self.pedestrian_angles = self._get_pedestrian_angles()
        self.pedestrian_directions = self._get_pedestrian_directions(self.pedestrian_angles)
        self.pedestrians += self.pedestrian_directions

        # Collision
        if (np.any(np.linalg.norm(self.car_front - self.pedestrians, axis=1) < 0.5) or
            np.any(np.linalg.norm(self.x - self.pedestrians, axis=1) < 0.5)):
            done = True
            reward -= (100*(2*self.speed)**2 + 0.5)

        if self.car_front[0] <= 0.0 or self.car_front[0] >= 9.0 or self.car_front[1] <= -5:
            done = True
            reward += -1000.0
        elif self.x[1] >= 4.0:
            reward += 250
            done = True

        return self.get_obs(), reward, done, dict(
                pedestrians=self.pedestrians,
                pedestrian_speeds = self.pedestrian_speeds,
                pedestrian_angles = self.pedestrian_angles)

    def _get_pedestrian_directions(self, angles):
        return self.pedestrian_speeds.reshape(-1,1) \
                * np.array([-np.sin(angles),
                            np.cos(angles)]).transpose()

    def get_obs(self):
        car_direction = self.car_front - self.x

        obs = np.concatenate([
            self.x.ravel(),
            self.car_front.ravel(),
            (self.car_front + car_direction).ravel(),
            [self.speed, self.angle],
            self.pedestrians.ravel(),
            (self.pedestrians + self.pedestrian_directions).ravel()])
        if self.use_vision:
            return dict(obs=obs, img=self._visualize(nparray=True))
        return obs

    def _visualize(self, show=False, filename=None, nparray=False, head_only=False, transparent=True):
        # if filename is None:
        #     plt.ion()
        #     self.fig = plt.figure()
        fig = self.fig
        fig.clf()

        # Draw boundaries
        #plt.plot([0, 9], [0, 0], linewidth=1, color='k')
        #plt.plot([0, 9], [4, 4], linewidth=1, color='k')
        plt.plot([0, 0], [-10, 10], linewidth=5, color='k')
        plt.plot([9, 9], [-10, 10], linewidth=5, color='k')

        plt.axis('square')
        plt.axis('off')
        plt.xlim((-1, 10))
        plt.ylim((-6, 5))
        #plt.xticks(np.arange(-1, 11))
        #plt.yticks(np.arange(-6, 6))

        # Car
        # Pose of the end of the car
        car = self.x.copy()
        if self.car is not None:
            self.car.remove()

        if not head_only:
            self.car = Rectangle((car[0], car[1]),
                        0.1, self.car_length, angle=np.rad2deg(self.angle), color=colors[-1])
        else:
            self.car = Circle([car[0], car[1]], radius=0.1, color=colors[-1], zorder=25)

        plt.gca().add_patch(self.car)

        # pedestrians
        for i in range(len(self.pedestrians)):
            ped = Circle(self.pedestrians[i], radius=0.2, color=colors[i], zorder=20)
            plt.gca().add_patch(ped)

            xy = self.pedestrians[i]
            dxdy = self.pedestrian_directions[i]

            rect = Rectangle((xy[0], xy[1]),
                    0.1, 1.0, angle=np.rad2deg(self.pedestrian_angles[i]), color=colors[i])
            plt.gca().add_patch(rect)
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
    env = CrossWalkEnv(use_vision=False)

    states = []
    actions = []


    obs = env.reset()
    states.append(dict(x=env.x.copy(), car_front=env.car_front.copy(), ped=env.pedestrians.copy()))

    print("obs", obs)
    done = False

    env._visualize(show=False, filename="imgs/crosswalk.png", head_only=True, transparent=False)
    for t in range(60):
        obs, rew, done, info = env.step(env.action_space.sample())
        env._visualize(show=False, filename="imgs/crosswalk_{}.png".format(t), head_only=(not done))
        if done:
            break


        print("obs", np.around(obs, 1))
        print("rew", rew)
        print("done", done)

