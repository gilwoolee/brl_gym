import numpy as np
from gym import utils
from brl_gym.envs.mujoco import mujoco_env
import os
from IPython import embed
from copy import copy
from gym import spaces
import cv2
import time
from math import sqrt

TARGET_TO_YPOS = {0:(2.5), 1:(-2.5)}

asset_dir = "/home/gilwoo/School_Workspace/bayesian_rl_bootstrap/brl_gym/brl_gym/envs/mujoco/"
class BoxPusher(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._max_episode_steps = 250*8
        self._elapsed_steps = 0
        self.target = 0
        high = np.array([1.0, 1.0, 1.0])
        self.action_weights = 1.0/high
        self.action_space = spaces.Box(low=-high, high=high)
        self.param_space = dict(
            ypos=spaces.Box(np.array([-2.5]),np.array([2.5]), dtype=np.float32))

        utils.EzPickle.__init__(self)

        self.action_space = spaces.Box(low=-high, high=high)
        self.cam_name = 'top_cam'
        self.frame_skip=1
        self.random_target = False
        self.reset()


    def step(self, a):
        # u_clipped = np.clip(a, self.action_space.low, self.action_space.high)

        # vec_1 = self.get_body_com("object") - self.get_body_com("pusher")
        # vec_2 = self.get_body_com("object") - self.get_body_com("goal")
        qpos = self.data.qpos
        object_pos_x = qpos[3]
        object_pos_y = qpos[4]
        target_pos_x = 2.0
        if self.target == 0:
            target_pos_y = self.param_space['ypos'].high
        else:
            target_pos_y = self.param_space['ypos'].low
        pusher_pos_x = qpos[0]
        pusher_pos_y = qpos[1]

        box_dist = sqrt((object_pos_x - pusher_pos_x)**2 + (object_pos_y - pusher_pos_y)**2)

        reward = 0
        if box_dist > 0.5:
         reward -= box_dist

        trgdist = sqrt((object_pos_x - target_pos_x)**2 + (object_pos_y - target_pos_y)**2)
        reward -= trgdist

        # self.data.ctrl[:] = a
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()

        # self._elapsed_steps += 1
        # if self._past_limit():
        #     ob = self.reset() # automatically reset the env
        dist_to_goal = sqrt((pusher_pos_x - target_pos_x)**2 + (pusher_pos_y - target_pos_y)**2)

        done = False

        reward /= 150.0
        return ob, reward, done, dict(goal_dist=dist_to_goal + np.random.normal() * 2.0)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset(self):
        if self.target == 0:
            self.fullpath = os.path.join(asset_dir, "assets", 'pusher_plane_2d.xml')
        else:
            self.fullpath = os.path.join(asset_dir, "assets", 'pusher_plane_2d_down.xml')
        mujoco_env.MujocoEnv.__init__(self, self.fullpath, 2)
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def reset_model(self):
        self._elapsed_steps = 0
        # pusher_position = np.array([-0.30, 0., 0.])
        # init_block_position = np.array([-0.15, 0., 0.025, 0.0])
        # goal_block_position = np.array([-0.15, 0.15])
        qpos = np.zeros(self.model.nq)
        # qpos[:3] = pusher_position
        # qpos[3:7] = init_block_position
        # qpos[7:9] = goal_block_position
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        if self.random_target:
            if np.random.rand() > 0.5:
                self.target = 1
            else:
                self.target = 0

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            # logger.debug("Env has passed the step limit defined by TimeLimit.")
            return True
        else:
            return False

    def get_state(self):
        return self.get_state_vector()

if __name__ == "__main__":
    env = BoxPusher()
    env.reset()
    print(env.get_state())
    # while True:
    #     a = env.action_space.sample()
    #     print(a)
    #     env.step(a)
    #     env.render()

