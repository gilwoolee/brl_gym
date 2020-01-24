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
        self.action_space = spaces.Box(np.array([-0.1, -0.5]), np.array([0.1, 0.5]))
        self.car_length = 1.0
        self.use_vision = use_vision

        if not use_vision:
            self.observation_space = []
        else:
            self.observation_space = spaces.Dict(dict(obs=[], img=[]))
        pass

    def reset(self):
        # Hidden intention of pedestrians
        self.goals = np.vstack([[9.0, 0.0], [np.random.choice(3, size=2) + 0.5]]).transpose()

        # Initial position of pedestrians (Each row is a pedestrian)
        self.pedestrians = np.vstack([[-0.5, 9.5], np.random.uniform(size=2)*4.0]).transpose()
        # Pedestrians have fixed speed, but can change directions
        self.pedestrian_speeds = np.ones(2)*0.25
        self.pedestrian_angles = self._get_pedestrian_angles()


        # Agent's initial position, speed, angle
        self.x = np.array([4.0,-5.0])
        self.speed = 0.5
        self.angle = 0.0

    def _get_pedestrian_angles(self):
        # Angles are directed straight to the goals
        diff = self.goals - self.pedestrians
        angles = -1.0*np.arctan2(diff[:,0], diff[:,1])
        angles += np.random.normal(size=2, scale=0.5)
        return angles

    def step(self, action):
        self.speed += action[0]
        self.angle += action[1]

        done = False
        reward = -0.1

        # move car
        self.x += self.speed*np.array([-np.sin(self.angle), np.cos(self.angle)])
        self.car_front = self.x + \
                         self.car_length * np.array([-np.sin(self.angle), np.cos(self.angle)])

        # move pedestrians
        if self.pedestrians[0,0] >= 9.0:
            self.pedestrian_speeds[0] = 0.0
        if self.pedestrians[1,0] <= 0.0:
            self.pedestrian_speeds[1] = 0.0

        self.pedestrian_angles = self._get_pedestrian_angles()
        self.pedestrian_directions = self._get_pedestrian_directions()
        self.pedestrians += self.pedestrian_directions

        # Collision
        if (np.any(np.linalg.norm(self.car_front - self.pedestrians, axis=1) < 0.5) or
            np.any(np.linalg.norm(self.x - self.pedestrians, axis=1) < 0.5)):
            done = True
            reward -= -(1000*self.speed**2 + 0.5)

        if self.car_front[0] <= 0.0 or self.car_front[0] >= 9.0:
            done = True
            reward += -1000.0
        elif self.x[1] >= 4.0:
            reward += 100
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
            car_direction.ravel(),
            self.pedestrians.ravel(),
            self.pedestrian_directions.ravel()])
        if self.use_vision:
            return dict(obs=obs, img=self._visualize(nparray=True))
        return obs

    def _visualize(self, show=False, filename=None, nparray=False):
        fig = plt.figure()

        # Draw boundaries
        plt.plot([0, 9], [0, 0], linewidth=1, color='k')
        plt.plot([0, 9], [4, 4], linewidth=1, color='k')
        plt.plot([0, 0], [-10, 10], linewidth=5, color='k')
        plt.plot([9, 9], [-10, 10], linewidth=5, color='k')

        # Car
        # Pose of the end of the car
        car = self.x.copy()
        car = Rectangle((car[0], car[1]),
                        0.1, 1.0, angle=np.rad2deg(self.angle), color='r')
        plt.gca().add_patch(car)

        # pedestrians
        for i in range(2):
            ped = Circle(self.pedestrians[i], radius=0.25, color='b', zorder=20)
            plt.gca().add_patch(ped)

        for i in range(2):
            goal = Circle(self.goals[i], radius=0.25, color='g', zorder=10)
            plt.gca().add_patch(goal)

        plt.axis('square')
        plt.xlim((-1, 10))
        plt.ylim((-6, 5))
        plt.xticks(np.arange(-1, 11))
        plt.yticks(np.arange(-6, 6))

        plt.grid()

        if np:
            return fig2np(fig)

        if filename is not None:
            plt.savefig(filename)

        if show:
            plt.show()



    def compute_reward(self):
        if self.x[1] > 4.0:
            # Done crossing
            return 0

        Rtime = -0.1

        # Check collision
        dist_to_peds = np.linalg.norm(self.x - self.pedestrians, axis=1)
        if np.any(dist_to_peds < 0.5):
            Rcol = -1000 * (self.speed**2 + 0.5)
            return Rtime + Rcol
        else:
            return Rtime

    def render(self):
        raise NotImplementedError

if __name__ == "__main__":
    env = CrossWalkEnv(use_vision=True)
    env.reset()
    env._visualize(show=True)
    obs, rew, done, info = env.step([1.0, -np.pi/4])
    import IPython; IPython.embed(); import sys; sys.exit(0)
    env._visualize(show=True)
    done = False
    while not done:
        obs, rew, done, info = env.step([0.0, 0.0])
        env._visualize(show=True)

        print("obs", np.around(obs, 1))
        print("rew", rew)
        print("done", done)
