import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from math import sqrt
from gym import spaces
import mujoco_py
import os

#TARGET_LOCATIONS = np.array([[2,-2], [2, 0],[2, 2], [0, 2], [-2, 2], [-2, 0], [-2,-2],[0, -2]]) * 0.2
TARGET_LOCATIONS = np.array([[2,-2], [2, 0],[2, 2], [1, 3],[1, -3]]) * 0.2

# TARGET_TO_YPOS = {0:(-0.2), 1:(0.2)}
asset_dir = "/home/gilwoo/School_Workspace/bayesian_rl_bootstrap/brl_gym/brl_gym/envs/mujoco/"


class Pusher(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)

        self.frame_skip = 20
        high = np.array([1, 1, 1])
        self.action_space = spaces.Box(low=-high, high=high)
        self.param_space = dict(
            ypos=spaces.Box(np.array([-0.2]),np.array([0.2]), dtype=np.float32))
        self.random_target = False
        self.target = 0
        self.reset()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        s = self.data.qpos
        object_pos_x = s[3]
        object_pos_y = s[4]

        target_pos_x, target_pos_y = TARGET_LOCATIONS[self.target]
        # if self.target == 0:
        #     target_pos_y = self.param_space['ypos'].low
        # else:
        #     target_pos_y = self.param_space['ypos'].high
        # target_pos_x = 0.2
        pusher_pos_x = s[0] - 0.08
        pusher_pos_y = s[1]
        ctrl = a

        box_dist = sqrt((object_pos_x - pusher_pos_x)**2 + (object_pos_y - pusher_pos_y)**2)

        trgdist = sqrt((object_pos_x - target_pos_x)**2 + (object_pos_y - target_pos_y)**2)
        reward = -trgdist + -box_dist - 0.01*sqrt(ctrl[0]**2 + ctrl[1]**2 + ctrl[2]**2)
        #dist_to_goal = sqrt((pusher_pos_y - target_pos_y)**2)# + (pusher_pos_y - target_pos_y)**2)
        # dist_to_goal = sqrt((object_pos_y - target_pos_y)**2 + (object_pos_x - target_pos_x)**2)

        done = False

        # if pusher_pos_x > target_pos_x :
        #     reward = 0.0
        #     done = True

        if trgdist < 0.05:
            reward = 1.0
            done = True
        return self._get_obs(), reward, done, dict(goal_dist=trgdist + np.random.normal(scale=1.0))

    def _get_obs(self):
        s = self.data.qpos
        object_pos_x = s[3]
        object_pos_y = s[4]
        pusher_pos_x = s[0] - 0.08
        pusher_pos_y = s[1]
        dists = TARGET_LOCATIONS - s[3:5]
        # dist_to_target_up = np.array([0.2, self.param_space['ypos'].high[0]]) - s[3:5]
        # dist_to_target_down = np.array([0.2, self.param_space['ypos'].low[0]]) - s[3:5]
        # print(dists)
        ob = np.concatenate([
            self.sim.data.qpos.flat[:-2],
            self.sim.data.qvel.flat[:-2],
            np.array([pusher_pos_x - object_pos_x])*-1,
            np.array([pusher_pos_y - object_pos_y])*-1,
            dists.flat
            # dist_to_target_down.flat,
            # dist_to_target_up.flat
        ])
        return ob

    def reset(self):
        # if self.target == 0:
        #     self.fullpath = os.path.join(asset_dir, "assets", 'pusher_down.xml')
        # else:
        self.fullpath = os.path.join(asset_dir, "assets", 'pusher.xml'.format(self.target))
        mujoco_env.MujocoEnv.__init__(self, self.fullpath, self.frame_skip)
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        init_q = np.concatenate([np.zeros(5), TARGET_LOCATIONS[self.target]])
        self.set_state(init_q, np.zeros(7))
        return self._get_obs()

    def get_state(self):
        return self._get_obs()

    def set_state(self, qpos, qvel):
        state = self.sim.get_state()
        for i in range(len(qpos)):
            state.qpos[i] = qpos[i]
        for i in range(len(qvel)):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()


if __name__ == "__main__":
    env = Pusher()

    o = env.reset()
    import IPython; IPython.embed()

    #while True:
    #    env.step(env.action_space.sample())
