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

def check_intersect(seg1, seg2):
    p0, p1 = seg1
    q0, q1 = seg2

    # Vectors
    u = p1 - p0
    v = q1 - q0

    # Vector perpendicular to v
    v_perp = np.array([-v[1], v[0]])

    # Vector q0 -> p0
    w = p0 - q0

    # Check parallel
    denom = np.dot(v_perp, u)
    if np.abs(denom) <= 1e-3:
        return False

    # Find s such that dot(v_perp, w + s*u) = 0
    s = -np.dot(v_perp,w)/denom

    # Intersection = w + s*u
    i = p0 + s * u
    if s < 0 or s > 1:
        return False

    # Find r for which i = q0 + r * v
    r = (i - q0)[0] / v[0]

    if r < 0 or r > 1:
        return False

    return True

def draw_and_check(seg1, seg2):
    check_intersect(seg1, seg2)

    seg1 = np.vstack(seg1)
    seg2 = np.vstack(seg2)

    plt.figure()
    plt.plot(seg1[:,0], seg1[:,1])
    plt.plot(seg2[:,0], seg2[:,1])

    plt.show()


# Crosswalk but with (velocity, steering angle) control
class CrossWalkVelEnv(gym.Env):
    def __init__(self, timestep=0.1):
        self.action_space = spaces.Box(np.array([0.0, -0.5]), np.array([1.2, 0.5]))
        self.car_length = 0.5 # Length of the MuSHR car
        self.num_pedestrians = 3
        self.timestep = timestep
        self.observation_space = spaces.Box(low=np.ones(20)*-10.0,
                                            high=np.ones(20)*10.0,
                                            dtype=np.float32)
        self.x_limit = np.array([0, 3.5], dtype=np.float32)
        self.y_limit = np.array([0, 4], dtype=np.float32)
        self.car_y_goal = 4.5
        self.x_left = 0.0
        self.x_right = 3.5
        self.y_starts = np.array([1.25, 3.7])
        self.ped_speed_limits = np.array([0.5, 0.8], dtype=np.float32)
        self.car_start_y = 0.0
        self.car_speed_limits = np.array([0.2, 0.4], dtype=np.float32)
        self.car_steering_limits = np.array([-0.5,0.5], dtype=np.float32)
        self.goal_xs = np.concatenate([np.ones(2)*self.x_right, np.ones(1)*self.x_left]).ravel()

    def _get_init_goal(self):
        while True:
            # Hidden intention of pedestrians
            goals = np.vstack([self.goal_xs, np.random.uniform(1.5, 3.5, size=3)]).transpose()

            # If too close, pass
            goals_close = False
            for pair in [(0,1), (0,2), (1,2)]:
                if np.abs(goals[pair[0],1] - goals[pair[1],1]) < 0.65:
                    goals_close = True
            if goals_close:
                continue

            # Initial position of pedestrians (Each row is a pedestrian)
            ped_left = np.vstack([self.x_left * np.ones(2), np.random.uniform(self.y_starts, size=2)]).transpose()
            ped_right = np.vstack([self.x_right * np.ones(1), np.random.uniform(self.y_starts[0], self.y_starts[1])]).transpose()

            # Check initial position distance
            if np.linalg.norm(ped_left[0] - ped_left[1]) < 0.5:
                continue

            pedestrians = np.vstack([ped_left, ped_right])

            # Check if paths cross
            intersect = False
            for pair in [(0,1), (0,2), (1,2)]:
                ped1 = (pedestrians[pair[0]], goals[pair[0]])
                ped2 = (pedestrians[pair[1]], goals[pair[1]])
                if check_intersect(ped1, ped2):
                    intersect = True
                    break

            if intersect:
                continue
            else:
                break

        return goals, pedestrians

    def reset(self):
        self.goals, self.pedestrians = self._get_init_goal()

        self.t = 0

        self.random_delays = np.random.choice(10, size=self.num_pedestrians)

        # Pedestrians have fixed speed, but can change directions
        self.pedestrian_speeds = np.random.uniform(
            self.ped_speed_limits[0], self.ped_speed_limits[1], size=self.num_pedestrians)
        self.pedestrian_angles = self._get_pedestrian_angles()

        # Agent's initial position, speed, angle
        self.pose = np.array([np.random.uniform(1.2, 2.8), 0.0, np.random.uniform(-0.5, 0.5)])
        self.speed = 0.0
        self.steering_angle = np.random.uniform(self.car_steering_limits[0], self.car_steering_limits[1])
        self.car_front = self.pose[:2] + \
                         self.car_length * np.array([-np.sin(self.pose[2]), np.cos(self.pose[2])])

        ped_speeds = self.pedestrian_speeds.copy()
        for i in range(2):
            if self.random_delays[i] < self.t:
                ped_speeds[i] = 0.0
        for i in [2]:
            if self.random_delays[i] < self.t:
                ped_speeds[i] = 0.0

        self.pedestrian_directions = self._get_pedestrian_directions(self.pedestrian_angles, ped_speeds)

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
        self.t += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.speed = action[0]
        theta = self.pose[2] + np.pi / 2.0

        # move car
        next_theta = theta + self.speed / self.car_length * np.tan(1e-5+action[1]) * self.timestep
        self.pose[0] += self.car_length / (1e-5+np.tan(action[1])) * (np.sin(next_theta) - np.sin(theta))
        self.pose[1] += self.car_length / (1e-5+np.tan(action[1])) * (-np.cos(next_theta) + np.cos(theta))
        self.pose[2] = next_theta - np.pi / 2.0
        self.car_front = self.pose[:2] + \
                         self.car_length * np.array([-np.sin(self.pose[2]), np.cos(self.pose[2])])

        # move pedestrians
        ped_speeds = self.pedestrian_speeds.copy()

        for i in range(2):
            if self.pedestrians[i,0] >= self.x_limit[1]:
                ped_speeds[i] = 0.0
            if self.random_delays[i] > self.t:
                ped_speeds[i] = 0.0

        for i in [2]:
            if self.random_delays[i] > self.t:
                ped_speeds[i] = 0.0
            if self.pedestrians[i,0] <= self.x_limit[0]:
                ped_speeds[i] = 0.0
        self.pedestrian_angles = self._get_pedestrian_angles()
        self.pedestrian_directions = self._get_pedestrian_directions(self.pedestrian_angles, ped_speeds)
        self.pedestrians += self.pedestrian_directions * self.timestep

        obs = self.get_obs()
        reward, done = self._get_reward(obs)
        return obs, reward, done, dict(
                pedestrians=self.pedestrians,
                pedestrian_speeds = ped_speeds,
                pedestrian_angles = self.pedestrian_angles)


    def _get_reward(self, obs):
        done = False
        pose = obs[:2]
        angle = obs[7]
        car_front = obs[2:4]
        speed = obs[6]
        pedestrians = obs[8:14].reshape(-1, 2)
        next_pedestrians = obs[14:20].reshape(-1,2)

        # Penalize time & distance to goal. y = 4.0
        reward = -0.1 - np.abs(self.car_y_goal - pose[1]) * 0.1

        # angle penalty
        reward -= np.abs(angle) * 0.5

        collision_dist = 0.65
        dist = np.linalg.norm(pose - pedestrians, axis=1)
        next_dist = np.linalg.norm(pose - next_pedestrians, axis=1)

        reward *= 0.05

        # Collision
        collision = False
        for i, (d, nd) in enumerate(zip(dist, next_dist)):
            if d < collision_dist and nd < d:
                collision = True
                break

        if collision:
            done = True
            reward -= 100
        elif car_front[0] <= self.x_limit[0] + 0.5\
                or car_front[0] >= self.x_limit[1] - 0.5\
                or car_front[1] <= -0.5:
            reward += -100.0
            done = True
        elif pose[1] >= self.car_y_goal:
            reward += 100.0
            done = True

        return reward, done

    def _get_pedestrian_directions(self, angles, ped_speeds):
        speeds = ped_speeds.reshape(-1,1)
        speeds += np.clip(np.random.normal(size=speeds.shape)*0.1, 0.0, 0.1)
        return speeds \
                * np.array([-np.sin(angles),
                            np.cos(angles)]).transpose()

    def get_obs(self):
        car_direction = self.car_front - self.pose[:2]

        obs = np.concatenate([
            self.pose[:2].ravel(), # 2
            self.car_front.ravel(), # 4
            (self.car_front + car_direction).ravel(), # 6
            [self.speed, self.pose[2]], # 8
            self.pedestrians.ravel(), # +3*2 = 14
            (self.pedestrians + self.pedestrian_directions).ravel()]) # + 3*2 = 20
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
        plt.plot([3.5, 3.5], [-1, 5], linewidth=2, color='k', linestyle='--')

        plt.axis('square')
        plt.axis('off')
        plt.xlim((-0.1, 3.6))
        plt.ylim((-0.5, self.car_y_goal+0.1))

        # Car
        # Pose of the end of the car
        car = self.pose.copy()
        angle = car[2]
        if self.car is not None:
            self.car.remove()

        if not head_only:
            self.car = Rectangle((car[0], car[1]),
                        0.33, self.car_length, angle=np.rad2deg(angle), color=colors[-1])
        else:
            self.car = Circle([car[0], car[1]], radius=0.5, color=colors[-1], zorder=25)

        plt.gca().add_patch(self.car)

        # pedestrians
        for i in range(len(self.pedestrians)):
            ped = Circle(self.pedestrians[i], radius=0.33, color=colors[i], zorder=20)
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
        plt.close()

    def _visualize_history(self, states, actions):
        pass

    def render(self, mode='human'):
        self._visualize(show=True)

if __name__ == "__main__":
    env = CrossWalkVelEnv()

    obs = env.reset()
    t = 0
    while True:
        t += 1
        o, r, d, _ = env.step([0.5, 0.1])
        env._visualize(filename="test{}.png".format(t))
        if d:
            break
        print(t)
