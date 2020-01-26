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
        self.pedestrian_speeds = np.clip(np.random.normal(scale=0.3, size=self.pedestrians.shape[0]) + 0.4, 0.1, 0.6)
        self.pedestrian_angles = self._get_pedestrian_angles()


        # Agent's initial position, speed, angle
        self.x = np.array([4.0,-5.0])
        self.speed = np.clip(np.random.normal(size=1) + 0.4, 0.3, 0.6)[0]
        self.angle = 0.0
        self.car_front = self.x + \
                         self.car_length * np.array([-np.sin(self.angle), np.cos(self.angle)])
        self.pedestrian_directions = self._get_pedestrian_directions()

        if self.fig is not None:
            self.fig.close()
        #self.fig = None
        self.car = None
        return self.get_obs()

    def _get_pedestrian_angles(self):
        # Angles are directed straight to the goals
        diff = self.goals - self.pedestrians
        angles = -1.0*np.arctan2(diff[:,0], diff[:,1])
        angles += np.random.normal(size=self.pedestrians.shape[0], scale=0.5)
        return angles

    def step(self, action):
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
        self.pedestrian_directions = self._get_pedestrian_directions()
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

    def _get_pedestrian_directions(self):
        return self.pedestrian_speeds.reshape(-1,1) \
                * np.array([-np.sin(self.pedestrian_angles),
                            np.cos(self.pedestrian_angles)]).transpose()

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

    def _visualize(self, show=False, filename=None, nparray=False):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure()
        fig = self.fig

        # Draw boundaries
        plt.plot([0, 9], [0, 0], linewidth=1, color='k')
        plt.plot([0, 9], [4, 4], linewidth=1, color='k')
        plt.plot([0, 0], [-10, 10], linewidth=5, color='k')
        plt.plot([9, 9], [-10, 10], linewidth=5, color='k')

        plt.axis('square')
        plt.xlim((-1, 10))
        plt.ylim((-6, 5))
        plt.xticks(np.arange(-1, 11))
        plt.yticks(np.arange(-6, 6))

        plt.grid()
        # Car
        # Pose of the end of the car
        car = self.x.copy()
        if self.car is not None:
            self.car.remove()
        self.car = Rectangle((car[0], car[1]),
                        0.1, self.car_length, angle=np.rad2deg(self.angle), color='r')
        plt.gca().add_patch(self.car)

        # pedestrians
        for i in range(len(self.pedestrians)):
            ped = Circle(self.pedestrians[i], radius=0.25, color='b', zorder=20)
            plt.gca().add_patch(ped)
        """
        for i in range(2):
            goal = Circle(self.goals[i], radius=0.25, color='g', zorder=10)
            plt.gca().add_patch(goal)
        """


        if nparray:
            raise NotImplementedError
            return fig2np(fig)

        if filename is not None:
            plt.savefig(filename)

        if show:
            fig.canvas.draw()

    def render(self, mode='human'):
        self._visualize(show=True)

if __name__ == "__main__":
    env = CrossWalkEnv(use_vision=False)
    obs = env.reset()
    print("obs", obs)

    env._visualize(show=True)
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        env._visualize(show=True)

        print("obs", np.around(obs, 1))
        print("rew", rew)
        print("done", done)

    #env._visualize(filename="test.png")
