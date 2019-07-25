import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

from gym import spaces
import time

from rllab.misc.overrides import overrides

# Modified from https://github.com/personalrobotics/nonprehensile-controllers/tree/master/envs
class PusherSingleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, max_episode_steps=None):
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        high = np.array([1.0,   0.1, 0.1])
        low  = np.array([-1.0, -0.1, -0.1])

        self.action_weights = 1.0 /(high - low)
        self.action_weights = self.action_weights / np.linalg.norm(self.action_weights)
        #self.action_weights = np.array([0.7, 0.1, 0.2])
        self.action_space = spaces.Box(low=low, high=high)
        self.max_interaction_time = 50
        self.fast_forward = True
        self.obs_error = .0

        utils.EzPickle.__init__(self)
        fullpath = os.path.join(os.path.dirname(__file__), "assets", 'pusher_single.xml')
        mujoco_env.MujocoEnv.__init__(self, fullpath, 2)
        self.cam_name = 'top_cam'
        self.frame_skip=5
        self.action_space = spaces.Box(low=low, high=high)

    def step(self, a):
        qpos = self.data.qpos
        qvel = self.data.qvel

        done = False
        # Timeout or beyond the line
        if self._elapsed_steps > self.max_interaction_time or qpos[0] > 0.0:
            # Just simulate without action
            if self.fast_forward:
                while np.linalg.norm(self.data.qvel) > 1e-5:
                    self.do_simulation(np.zeros(self.model.nu), self.frame_skip)
                    self._elapsed_steps += 1
                #self.data.qvel[:] = 0
                done = True
            else:
                self.do_simulation(np.zeros(self.model.nu), self.frame_skip)
                self._elapsed_steps += 1

            #print('r', np.around(self.data.qpos[:3],2), np.around(self.get_body_com('object'),2))
        else:
            #print(a)
            u_clipped = np.clip(a, self.action_space.low, self.action_space.high)
            self.data.qvel[:3] = u_clipped
            self.do_simulation(np.zeros(self.model.nu), self.frame_skip)
            #self.do_simulation(u_clipped, self.frame_skip)
            self._elapsed_steps += 1

        obj_to_goal  = self.get_body_com("object") - self.get_body_com("goal")

        reward = - np.linalg.norm(obj_to_goal[:2])

        obj = self.get_body_com('object')
        if obj[0] <= 0.0 or self._elapsed_steps < self.max_interaction_time:
            reward_ctrl = - np.square(a*self.action_weights).sum()
            reward += 0.1 * reward_ctrl

        ob = self._get_obs()

        if self._past_limit():
            if self.metadata.get('semantics.autoreset'): # This should usually be false
                raise NotImplementedError('CHECK WHY IS AUTORESET SET TO TRUE')
                _ = self.reset() # automatically reset the env
            done = True

        if self._elapsed_steps > self.max_interaction_time and np.linalg.norm(self.data.qvel) < 1e-5:
            done = True

        return ob, reward, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 15.0

    def reset_model(self):
        self._elapsed_steps = 0
        init_position = np.array([-0.25, 0, 0, 0])
        goal_position = np.array([1])
        bgoal_position = np.array([-1])

        qpos = np.zeros(self.model.nq)
        qpos[0:4] = init_position
        qpos[4] = goal_position
        qpos[5] = bgoal_position

        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        for _ in range(30):
            self.do_simulation(np.zeros(self.model.nu), self.frame_skip)
        return self._get_obs()

    def _get_obs(self):
        if  np.random.random() > self.obs_error:
            pos = np.concatenate([
                self.sim.data.qpos.flat[:4],
                self.get_body_com('goal')[:2]])
        else:
            pos = np.concatenate([
                self.sim.data.qpos.flat[:4],
                self.get_body_com('bad_goal')[:2]])

        return np.concatenate([
            pos,
            self.sim.data.qvel.flat[:3],
        ])

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            # logger.debug("Env has passed the step limit defined by TimeLimit.")
            return True
        else:
            return False

    @overrides
    def log_diagnostics(self, paths):
        pass

    def terminate(self):
        pass

